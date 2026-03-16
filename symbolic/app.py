import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import markdown
from config import (BASE_DIR, EMBEDDER_MODEL_DIR, EMBEDDER_MODEL_NAME,
                    FALLBACK_MODEL_DIR, FALLBACK_MODEL_NAME, HF_TOKEN)
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from manager.dialogue_manager import handle_dialogue
from models.model_loader import load_embedder, load_fallback_model
from rag.rag_manager import (add_docs, chroma_initialized,
                             load_game_docs_from_disk, set_embedder)
from schemas import AskReq, AskRes

templates = Jinja2Templates(directory="templates")
model_ready = False

async def load_models(app: FastAPI):
    global model_ready
    print("🚀 starting model loading...")
    fb_tokenizer, fb_model = load_fallback_model(FALLBACK_MODEL_NAME, FALLBACK_MODEL_DIR, token=HF_TOKEN)
    app.state.fallback_tokenizer = fb_tokenizer
    app.state.fallback_model = fb_model

    embedder = load_embedder(EMBEDDER_MODEL_NAME, EMBEDDER_MODEL_DIR, token=HF_TOKEN)
    app.state.embedder = embedder
    set_embedder(embedder)

    docs_path = BASE_DIR / "rag" / "docs"
    if not chroma_initialized():
        docs = load_game_docs_from_disk(str(docs_path))
        add_docs(docs)
        print(f"✅ finished inserting {len(docs)} documents into RAG DB")
    else:
        print("🔄 already initialized RAG DB")

    model_ready = True
    print("✅ model loading complete, server is ready to accept requests")

@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(load_models(app))
    yield
    print("🛑 shutting down...")

app = FastAPI(title="neuro-engine", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://fpsgame-rrbc.onrender.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", include_in_schema=False)
async def root(request: Request):
    md_path = Path(__file__).parent / "README.md"
    md_content = md_path.read_text(encoding="utf-8")

    start_tag = "<!-- app-tab:start -->"
    end_tag = "<!-- app-tab:end -->"
    if start_tag in md_content and end_tag in md_content:
        short_md = md_content.split(start_tag)[1].split(end_tag)[0].strip()
    else:
        short_md = md_content  # fallback: all content if tags not found

    html_from_md = markdown.markdown(short_md)
    return templates.TemplateResponse("index.html", {"request": request, "readme_content": html_from_md})

@app.get("/status")
async def status():
    return {"ready": model_ready}

@app.post("/wake")
async def wake(request: Request):
    session_id = (await request.json()).get("session_id", "unknown")
    print(f"📡 Wake signal received for session: {session_id}")
    if not model_ready:
        asyncio.create_task(load_models(app))
    return {"status": "awake", "model_ready": model_ready}

@app.post("/ask", response_model=AskRes)
async def ask(request: Request, req: AskReq):
    if not model_ready:
        raise HTTPException(status_code=503, detail="Model not ready")
    if not req.context:
        raise HTTPException(status_code=400, detail="missing context")
    if not (req.session_id and req.npc_id and req.user_input):
        raise HTTPException(status_code=400, detail="missing fields")

    context = req.context
    npc_config_dict = context.npc_config.model_dump() if context.npc_config else None

    return await handle_dialogue(
        request=request,
        session_id=req.session_id,
        npc_id=req.npc_id,
        user_input=req.user_input,
        context=context.model_dump(),
        npc_config=npc_config_dict
    )

