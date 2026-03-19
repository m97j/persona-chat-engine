from inference import run_inference
from modules.ui_components import build_ui
from webtest_prompt import build_webtest_prompt


# Web Test UI Call Function
def gradio_infer(npc_id, npc_location, player_utt):
    prompt = build_webtest_prompt(npc_id, npc_location, player_utt)
    result = run_inference(prompt)
    return result["npc_output_text"], result["deltas"], result["flags_prob"]

# ping: Check status and wake up
def ping():
    # Check if model is loaded, load if not
    global wrapper, tokenizer, model, flags_order
    if 'model' not in globals() or model is None:
        from model_loader import ModelWrapper
        wrapper = ModelWrapper()
        tokenizer, model, flags_order = wrapper.get()
    return {"status": "awake"}


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860)
