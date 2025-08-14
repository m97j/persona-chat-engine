from typing import Dict, Any, Tuple
from .utils.model_loader import hf_has_model
from .utils.hf_client import call_main
from .config import GEN_DEFAULTS

async def generate(session_id: str, npc_id: str, prompt: str, gen_params: Dict[str, Any] | None = None) -> Tuple[str, Dict[str, Any]]:
    if not await hf_has_model("main"):
        raise RuntimeError("HF main model not available")
    payload = {
        "session_id": session_id,
        "npc_id": npc_id,
        "prompt": prompt,
        "gen": gen_params or GEN_DEFAULTS
    }
    resp = await call_main(payload)
    return resp.get("text", ""), resp.get("meta", {})