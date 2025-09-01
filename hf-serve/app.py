import gradio as gr
from inference import run_inference, reload_model  # reload_model은 모델 재로딩 함수
from utils_prompt import build_webtest_prompt

# UI에서 호출할 함수
def gradio_infer(npc_id, npc_location, player_utt):
    prompt = build_webtest_prompt(npc_id, npc_location, player_utt)
    result = run_inference(prompt)
    return result["npc_output_text"], result["deltas"], result["flags_prob"]

# API 호출용 함수
def api_infer(session_id, npc_id, prompt, max_tokens=200):
    result = run_inference(prompt)
    return {
        "session_id": session_id,
        "npc_id": npc_id,
        "npc_response": result["npc_output_text"],
        "deltas": result["deltas"],
        "flags": result["flags_prob"],
        "thresholds": result["flags_thr"]
    }

# 모델 재로딩용 함수
def ping_reload():
    reload_model(branch="latest")  # latest 브랜치에서 재다운로드 & 로드
    return {"status": "reloaded"}

with gr.Blocks() as demo:
    gr.Markdown("## NPC Main Model Inference")

    with gr.Tab("Web Test UI"):
        npc_id = gr.Textbox(label="NPC ID")
        npc_loc = gr.Textbox(label="NPC Location")
        player_utt = gr.Textbox(label="Player Utterance")
        npc_resp = gr.Textbox(label="NPC Response")
        deltas = gr.JSON(label="Deltas")
        flags = gr.JSON(label="Flags Probabilities")
        btn = gr.Button("Run Inference")

        # UI 버튼 클릭 시 API 엔드포인트도 자동 생성
        btn.click(
            fn=gradio_infer,
            inputs=[npc_id, npc_loc, player_utt],
            outputs=[npc_resp, deltas, flags],
            api_name="predict_main"  # /api/predict_main 엔드포인트 생성
        )

    # 별도의 UI 없이 API만 제공하는 엔드포인트
    gr.Button("Reload Model").click(
        fn=ping_reload,
        inputs=[],
        outputs=[],
        api_name="ping_reload"  # /api/ping_reload 엔드포인트 생성
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)  
