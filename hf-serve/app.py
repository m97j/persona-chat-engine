from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr
from inference import run_inference
from utils_prompt import build_prompt

app = FastAPI()

class PromptRequest(BaseModel):
    session_id: str
    npc_id: str
    prompt: str
    max_tokens: int = 200

@app.post("/wake/")
async def wake():
    return {"status": "ready"}

@app.post("/predict_main/")
async def predict_main(req: PromptRequest):
    result = run_inference(req.prompt)
    return result

# Gradio Blocks
def gradio_infer(npc_id, npc_location, player_utt):
    pre = {
        "npc_id": npc_id,
        "npc_location": npc_location,
        "tags": {},
        "lore": "",
        "description": "",
        "player_state": {},
        "context": [],
        "player_utterance": player_utt
    }
    prompt = build_prompt(pre)
    result = run_inference(prompt)
    return result["npc_output_text"], result["deltas"], result["flags_prob"]

demo = gr.Interface(
    fn=gradio_infer,
    inputs=[
        gr.Textbox(label="NPC ID"),
        gr.Textbox(label="NPC Location"),
        gr.Textbox(label="Player Utterance")
    ],
    outputs=[
        gr.Textbox(label="NPC Response"),
        gr.JSON(label="Deltas"),
        gr.JSON(label="Flags Probabilities")
    ],
    title="NPC Main Model Inference"
)

app = gr.mount_gradio_app(app, demo, path="/ui")
