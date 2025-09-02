import gradio as gr
from .case_loader import get_case_names, load_case, run_case

def build_ui():
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="purple")) as demo:
        # 상단 소개
        gr.Markdown("""
        # 👾 PersonaChatEngine HF-Serve
        **게임 내 NPC 메인 모델 추론 서버**  
        Qwen 3B 기반 LoRA 파인튜닝 모델을 사용하여 NPC 대사를 생성합니다.
        """)

        with gr.Row():
            gr.Button("📄 상세 문서 보기",
                      link="https://huggingface.co/spaces/m97j/PersonaChatEngine_HF-serve/blob/main/README.md")
            gr.Button("💻 Colab 테스트 열기",
                      link="https://colab.research.google.com/drive/1_-qH8kdoU2Jj58TdaSnswHex-BFefInq?usp=sharing#scrollTo=cFJGv8BJ8oPD")

        gr.Markdown("### 🎯 테스트 케이스 기반 간단 실행")

        with gr.Row():
            case_dropdown = gr.Dropdown(choices=get_case_names(), label="테스트 케이스 선택", value=get_case_names()[0])
            load_btn = gr.Button("케이스 불러오기")

        case_info = gr.Textbox(label="케이스 정보", lines=10)
        player_input = gr.Textbox(label="Player Utterance 수정", lines=2)

        run_btn = gr.Button("🚀 Run Inference", variant="primary")
        npc_resp = gr.Textbox(label="NPC Response")
        deltas = gr.JSON(label="Deltas")
        flags = gr.JSON(label="Flags Probabilities")

        load_btn.click(
            fn=lambda name: load_case(get_case_names().index(name)),
            inputs=[case_dropdown],
            outputs=[case_info, player_input]
        )

        run_btn.click(
            fn=lambda name, utt: run_case(get_case_names().index(name), utt),
            inputs=[case_dropdown, player_input],
            outputs=[npc_resp, deltas, flags]
        )

        gr.Markdown("""
        ---
        ⚠️ **실제 게임 파이프라인 테스트**는 [ai-server Swagger](https://huggingface.co/spaces/m97j/PersonaChatEngine_ai_server)에서 진행하세요.
        """)

    return demo
