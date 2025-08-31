import torch
from .config import DEVICE, MAX_LENGTH
from .model_loader import ModelWrapper

# 전역 로드 (서버 시작 시 1회)
wrapper = ModelWrapper()
tokenizer, model, flags_order = wrapper.get()

GEN_PARAMS = {
    "max_new_tokens": 200,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True,
    "repetition_penalty": 1.05
}

def run_inference(prompt: str):
    # 토크나이즈
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(DEVICE)

    with torch.no_grad():
        # LM 생성
        gen_ids = model.generate(**inputs, **GEN_PARAMS)
        generated_text = tokenizer.decode(gen_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # 히든 상태 추출
        outputs = model(**inputs, output_hidden_states=True)
        h = outputs.hidden_states[-1]

        # <STATE> 토큰 위치 pooling
        STATE_ID = tokenizer.convert_tokens_to_ids("<STATE>")
        ids = inputs["input_ids"]
        mask = (ids == STATE_ID).unsqueeze(-1)
        if mask.any():
            counts = mask.sum(dim=1).clamp_min(1)
            pooled = (h * mask).sum(dim=1) / counts
        else:
            pooled = h[:, -1, :]  # fallback: 마지막 토큰

        # heads 예측
        delta_pred = torch.tanh(model.delta_head(pooled))[0].cpu().tolist()
        flag_prob = torch.sigmoid(model.flag_head(pooled))[0].cpu().tolist()
        flag_thr = torch.sigmoid(model.flag_threshold_head(pooled))[0].cpu().tolist()

    # index→name 매핑
    flags_prob_dict = {name: round(prob, 6) for name, prob in zip(flags_order, flag_prob)}
    flags_thr_dict = {name: round(thr, 6) for name, thr in zip(flags_order, flag_thr)}

    return {
        "npc_output_text": generated_text.strip(),
        "deltas": {
            "trust": delta_pred[0],
            "relationship": delta_pred[1]
        },
        "flags_prob": flags_prob_dict,
        "flags_thr": flags_thr_dict
    }
