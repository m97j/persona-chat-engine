import torch
from config import (DEVICE, GEN_MAX_NEW_TOKENS, GEN_TEMPERATURE, GEN_TOP_P,
                    MAX_LENGTH)
from model_loader import ModelWrapper

# Global Load (once at server start)
wrapper = ModelWrapper()
tokenizer, model, flags_order = wrapper.get()

GEN_PARAMS = {
    "max_new_tokens": GEN_MAX_NEW_TOKENS,
    "temperature": GEN_TEMPERATURE,
    "top_p": GEN_TOP_P,
    "do_sample": True,
    "repetition_penalty": 1.05,
}

def run_inference(prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(DEVICE)

    with torch.no_grad():
        # language generation
        gen_ids = model.generate(**inputs, **GEN_PARAMS)
        generated_text = tokenizer.decode(
            gen_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

        # hidden state extraction
        outputs = model(**inputs, output_hidden_states=True)
        h = outputs.hidden_states[-1]

        # <STATE> token position pooling
        STATE_ID = tokenizer.convert_tokens_to_ids("<STATE>")
        ids = inputs["input_ids"]
        mask = (ids == STATE_ID).unsqueeze(-1)
        if mask.any():
            counts = mask.sum(dim=1).clamp_min(1)
            pooled = (h * mask).sum(dim=1) / counts
        else:
            pooled = h[:, -1, :]

        # delta, flag, flag_threshold prediction
        delta_pred = torch.tanh(model.delta_head(pooled))[0].cpu().tolist()
        flag_prob = torch.sigmoid(model.flag_head(pooled))[0].cpu().tolist()
        flag_thr = torch.sigmoid(model.flag_threshold_head(pooled))[0].cpu().tolist()

    flags_prob_dict = {name: round(prob, 6) for name, prob in zip(flags_order, flag_prob)}
    flags_thr_dict = {name: round(thr, 6) for name, thr in zip(flags_order, flag_thr)}

    return {
        "npc_output_text": generated_text.strip(),
        "deltas": {
            "trust": float(delta_pred[0]),
            "relationship": float(delta_pred[1]),
        },
        "flags_prob": flags_prob_dict,
        "flags_thr": flags_thr_dict,
    }

def reload_model(branch="latest"):
    global wrapper, tokenizer, model, flags_order
    wrapper = ModelWrapper(branch=branch)
    tokenizer, model, flags_order = wrapper.get()
    print(f"Model reloaded from branch: {branch}")
