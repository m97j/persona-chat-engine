from transformers import pipeline
from fastapi import Request

async def detect_emotion(request: Request, text: str) -> dict:
    tokenizer = request.app.state.emotion_tokenizer
    model = request.app.state.emotion_model

    emotion_pipeline = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        return_all_scores=True
    )

    results = emotion_pipeline(text)
    # 결과를 label: score 형태로 변환
    return {r["label"]: r["score"] for r in results[0]}