# ai-server/app.py
import os, json, logging, asyncio
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai-server")

HF_SERVE_BASE = os.getenv("LOCAL_HF_URL", "http://hf-serve:5000")
HF_TIMEOUT = float(os.getenv("HF_TIMEOUT", "20"))

app = FastAPI(title="ai-server")

# simple config load
with open("npc_config.json", "r", encoding="utf-8") as f:
    NPCS = json.load(f)

# helper to call local hf-serve
async def call_hf(endpoint: str, payload: dict):
    url = HF_SERVE_BASE.rstrip("/") + endpoint
    async with httpx.AsyncClient(timeout=HF_TIMEOUT) as client:
        r = await client.post(url, json=payload)
        r.raise_for_status()
        return r.json()

# preprocess: rule-based (optionally extend)
async def preprocess(session_id, npc_id, text, context):
    # quick rule filter (mirrors hf-serve fallback)
    triggers = {"grief": ["아이", "죽", "사라졌", "잃"], "anger": ["죽여", "복수"]}
    for k, kws in triggers.items():
        if any(kw in text for kw in kws):
            return {"trigger": k, "source": "rule", "confidence": 0.95}
    # if local hf-serve has preprocess model, call it
    try:
        resp = await call_hf("/predict_preprocess", {"session_id": session_id, "npc_id": npc_id, "text": text, "context": context})
        return resp
    except Exception as e:
        logger.info("preprocess hf call failed or not present: %s", e)
        return {"trigger": None, "confidence": 0.1, "source": "fallback"}

# postprocess: call hf-serve postprocess or apply simple rules
async def postprocess(session_id, npc_id, generated_text, constraints):
    try:
        resp = await call_hf("/predict_postprocess", {"session_id": session_id, "npc_id": npc_id, "text": generated_text, "constraints": constraints})
        return resp
    except Exception as e:
        logger.info("postprocess hf call failed or not present: %s", e)
        # local simple cleaner
        if constraints.get("no_violence") and ("죽여" in generated_text or "kill" in generated_text):
            cleaned = generated_text.replace("죽여", "[삭제]").replace("kill", "[redacted]")
            return {"text": cleaned, "valid": False, "meta": {"cleaned": True}}
        return {"text": generated_text, "valid": True, "meta": {}}

@app.post("/ask")
async def ask(request: Request):
    body = await request.json()
    session_id = body.get("session_id")
    npc_id = body.get("npc_id")
    user_input = body.get("user_input")
    context = body.get("context", {})

    if not (session_id and npc_id and user_input):
        raise HTTPException(status_code=400, detail="missing fields")
    if npc_id not in NPCS:
        raise HTTPException(status_code=404, detail="unknown npc")

    # 1) Preprocess rule/classify
    pre = await preprocess(session_id, npc_id, user_input, context)

    # Option: if preprocess indicates immediate behavior (ignore/attack), return early
    if pre.get("source") == "rule" and pre.get("trigger") == "ignore":
        return {"npc_response": "...", "flags": {"trigger": "ignored"}}

    # 2) Build prompt (can be enriched with short history, RAG results, persona)
    persona = NPCS[npc_id]
    system = f"Persona: {persona.get('persona_name')} - {persona.get('description')}"
    prompt = f"{system}\nContext: {context}\nPreprocess: {pre}\nPlayer: {user_input}\nNPC:"

    # 3) Call hf-serve main generate (prefer local main)
    try:
        main_resp = await call_hf("/predict_main", {"session_id": session_id, "npc_id": npc_id, "prompt": prompt, "max_tokens": 200})
        generated = main_resp.get("text") if isinstance(main_resp, dict) else str(main_resp)
    except Exception as e:
        logger.exception("Failed to call /predict_main: %s", e)
        # fallback simple response
        generated = "..." 

    # 4) postprocess (filter / rewrite)
    constraints = {"no_violence": True}
    post = await postprocess(session_id, npc_id, generated, constraints)

    # 5) Decide flags (example: propagate preprocess trigger)
    flags = {}
    if pre.get("trigger"):
        flags["trigger"] = pre["trigger"]

    # Optionally: return delta values (emotion_delta etc.) in 'meta' for game-server to accumulate
    meta = post.get("meta", {})
    valid = post.get("valid", True)
    npc_response = post.get("text", generated)

    return {"npc_response": npc_response, "flags": flags, "valid": valid, "meta": meta}
