from utils.hf_client import call_main

async def generate_response(session_id: str, npc_id: str, prompt: str, max_tokens: int = 200) -> dict:
    payload = {
        "session_id": session_id,
        "npc_id": npc_id,
        "prompt": prompt,
        "max_tokens": max_tokens
    }
    return await call_main(payload)  # {"text":..., "delta":..., "flag":...}
