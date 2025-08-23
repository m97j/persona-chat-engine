#from typing import Dict, Any, Tuple
from utils.hf_client import call_main
from manager.prompt_builder import build_prompt
from rag.rag_generator import retrieve

async def generate_response(session_id: str, npc_id: str, preprocessed: dict) -> str:
    docs = retrieve(preprocessed["text"], preprocessed["npc_config"]["id"])
    prompt = build_prompt(preprocessed, docs)

    payload = {
        "session_id": session_id,
        "npc_id": npc_id,
        "prompt": prompt,
        "max_tokens": 200
    }
    result = await call_main(payload)
    return result.get("text", "...")




