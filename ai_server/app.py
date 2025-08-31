from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from manager.dialogue_manager import handle_dialogue
from rag.rag_generator import chroma_initialized, load_game_docs_from_disk, add_docs
from contextlib import asynccontextmanager
from models.model_loader import load_emotion_model, load_fallback_model, load_embedder
from schemas import AskReq, AskRes
from pathlib import Path
from rag.rag_generator import set_embedder

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
    set_embedder(embedder)  # 추가

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
최종 game‑server → ai‑server 요청 예시
{
  "session_id": "abc123",
  "npc_id": "mother_abandoned_factory",
  "user_input": "아! 머리가… 기억이 떠올랐어요.",

  /* game-server에서 필터링한 필수/선택 require 요소만 포함 */
  "context": {
    "require": {
      "items": ["photo_forgotten_party"],       // 필수/선택 구분은 npc_config.json에서
      "actions": ["visited_factory"],
      "game_state": ["box_opened"],             // 필요 시
      "delta": { "trust": 0.35, "relationship": 0.1 }
    },

    "player_state": {
      "level": 7,
      "reputation": "helpful",
      "location": "map1"
      /* 전체 인벤토리/행동 로그는 필요 시 별도 전달 */
    },

    "game_state": {
      "current_quest": "search_jason",
      "quest_stage": "in_progress",
      "location": "map1",
      "time_of_day": "evening"
    },

    "npc_state": {
      "id": "mother_abandoned_factory",
      "name": "실비아",
      "persona_name": "Silvia",
      "dialogue_style": "emotional",
      "relationship": 0.35,
      "npc_mood": "grief"
    },

    "dialogue_history": [
      {
        "player": "혹시 이 공장에서 본 걸 말해줘요.",
        "npc": "그날을 떠올리는 게 너무 힘들어요."
      }
    ]
  }
}
'''

'''
{
  "session_id": "abc123",
  "npc_id": "mother_abandoned_factory",
  "user_input": "아! 머리가… 기억이 떠올랐어요.",
  "precheck_passed": true,
  "context": {
    "player_status": {
      "level": 7,
      "reputation": "helpful",
      "location": "map1",

      "trigger_items": ["photo_forgotten_party"],   // game-server에서 조건 필터 후 key로 변환
      "trigger_actions": ["visited_factory"]        // 마찬가지로 key 문자열

      /* 원본 전체 inventory/actions 배열은 서비스 필요 시 별도 전달 가능
         하지만 ai-server 조건 판정에는 trigger_*만 사용 */
    },
    "game_state": {
      "current_quest": "search_jason",
      "quest_stage": "in_progress",
      "location": "map1",
      "time_of_day": "evening"
    },
    "npc_config": {
      "id": "mother_abandoned_factory",
      "name": "실비아",
      "persona_name": "Silvia",
      "dialogue_style": "emotional",
      "relationship": 0.35,
      "npc_mood": "grief",
      "trigger_values": {
        "in_progress": ["기억", "사진", "파티"]
      },
      "trigger_definitions": {
        "in_progress": {
          "required_text": ["기억", "사진"],
          "required_items": ["photo_forgotten_party"], // trigger_items와 매칭
          "required_actions": ["visited_factory"],     // trigger_actions와 매칭
          "emotion_threshold": { "sad": 0.2 },
          "fallback_style": {
            "style": "guarded",
            "npc_emotion": "suspicious"
          }
        }
      }
    },
    "dialogue_history": [
      {
        "player": "혹시 이 공장에서 본 걸 말해줘요.",
        "npc": "그날을 떠올리는 게 너무 힘들어요."
      }
    ]
  }
}

------------------------------------------------------------------------------------------------------

이전 game-server 요청 구조 예시:
{
  "session_id": "abc123",
  "npc_id": "mother_abandoned_factory",
  "user_input": "아! 머리가… 기억이 떠올랐어요.",
  "context": {
    "player_status": {
      "level": 7,
      "reputation": "helpful",
      "location": "map1",
      "items": ["photo_forgotten_party"],
      "actions": ["visited_factory", "talked_to_guard"]
    },
    "game_state": {
      "current_quest": "search_jason",
      "quest_stage": "in_progress",
      "location": "map1",
      "time_of_day": "evening"
    },
    "npc_config": {
      "id": "mother_abandoned_factory",
      "name": "실비아",
      "persona_name": "Silvia",
      "dialogue_style": "emotional",
      "relationship": 0.35,
      "npc_mood": "grief",
      "trigger_values": {
        "in_progress": ["기억", "사진", "파티"]
      },
      "trigger_definitions": {
        "in_progress": {
          "required_text": ["기억", "사진"],
          "emotion_threshold": {"sad": 0.2},
          "fallback_style": {"style": "guarded", "npc_emotion": "suspicious"}
        }
      }
    },
    "dialogue_history": [
      {"player": "혹시 이 공장에서 본 걸 말해줘요.", "npc": "그날을 떠올리는 게 너무 힘들어요."}
    ]
  }
}

'''