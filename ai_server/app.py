from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from manager.dialogue_manager import handle_dialogue
from rag.rag_generator import chroma_initialized, load_game_docs_from_disk, add_docs
from contextlib import asynccontextmanager
from models.model_loader import load_emotion_model, load_fallback_model, load_embedder
from schemas import AskReq, AskRes
from pathlib import Path


# 모델 이름
EMOTION_MODEL_NAME = "tae898/emoberta-base-ko"
FALLBACK_MODEL_NAME = "skt/ko-gpt-trinity-1.2B-v0.5"
EMBEDDER_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# 절대 경로 기준 모델 디렉토리 설정
BASE_DIR = Path(__file__).resolve().parent  # ai_server/
EMOTION_MODEL_DIR = BASE_DIR / "models" / "emotion-classification-model"
FALLBACK_MODEL_DIR = BASE_DIR / "models" / "fallback-npc-model"
EMBEDDER_MODEL_DIR = BASE_DIR / "models" / "sentence-embedder"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Emotion
    emo_tokenizer, emo_model = load_emotion_model(EMOTION_MODEL_NAME, EMOTION_MODEL_DIR)
    app.state.emotion_tokenizer = emo_tokenizer
    app.state.emotion_model = emo_model

    # Fallback
    fb_tokenizer, fb_model = load_fallback_model(FALLBACK_MODEL_NAME, FALLBACK_MODEL_DIR)
    app.state.fallback_tokenizer = fb_tokenizer
    app.state.fallback_model = fb_model

    # Embedder
    embedder = load_embedder(EMBEDDER_MODEL_NAME, EMBEDDER_MODEL_DIR)
    app.state.embedder = embedder

    print("✅ 모든 모델 로딩 완료")

    # RAG 초기화
    docs_path = BASE_DIR / "rag" / "docs"
    if not chroma_initialized():
        docs = load_game_docs_from_disk(str(docs_path))
        add_docs(docs)
        print(f"✅ RAG 문서 {len(docs)}개 삽입 완료")
    else:
        print("🔄 RAG DB 이미 초기화됨")

    yield  # 앱 실행

    print("🛑 서버 종료 중...")


app = FastAPI(title="ai-server", lifespan=lifespan)

# CORS 설정 (game-server에서 요청 가능하도록)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://fpsgame-rrbc.onrender.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/ask", response_model=AskRes)
async def ask(request: Request, req: AskReq):
    context = req.context or {}
    npc_config = context.npc_config

    if not (req.session_id and req.npc_id and req.user_input and npc_config):
        raise HTTPException(status_code=400, detail="missing fields")

    result = await handle_dialogue(
        request=request,
        session_id=req.session_id,
        npc_id=req.npc_id,
        user_input=req.user_input,
        context=context.dict(),
        npc_config=npc_config.dict()
    )
    return result


@app.post("/wake")
async def wake(request: Request):
    body = await request.json()
    session_id = body.get("session_id", "unknown")
    print(f"📡 Wake signal received for session: {session_id}")
    return {"status": "awake", "session_id": session_id}


'''
game-server 요청 구조 예시:
{
  "session_id": "abc123",
  "npc_id": "npc_001",
  "user_input": "안녕, 오늘 기분 어때?",
  "context": {
    "player_status": {
      "level": 12,
      "inventory": ["sword", "potion"],
      "reputation": "neutral"
    },
    "game_state": {
      "current_quest": "dragon_hunt",
      "location": "village",
      "time_of_day": "evening"
    },
    "npc_config": {
      "name": "엘라",
      "personality": "친절하고 조용함",
      "backstory": "마을의 약초상으로, 과거에 용병이었던 경험이 있음",
      "dialogue_style": "짧고 단정한 말투",
      "relationship": "친구"
    }
  }
}
'''