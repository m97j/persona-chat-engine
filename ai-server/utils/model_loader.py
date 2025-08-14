import httpx
from ..config import LOCAL_HF_BASE, HF_TIMEOUT

async def hf_health() -> bool:
    try:
        async with httpx.AsyncClient(timeout=HF_TIMEOUT) as client:
            r = await client.get(f"{LOCAL_HF_BASE.rstrip('/')}/health")
            return r.status_code == 200
    except Exception:
        return False

async def hf_has_model(kind: str) -> bool:
    try:
        async with httpx.AsyncClient(timeout=HF_TIMEOUT) as client:
            r = await client.get(f"{LOCAL_HF_BASE.rstrip('/')}/has_model/{kind}")
            if r.status_code == 200:
                j = r.json()
                return bool(j.get("exists"))
    except Exception:
        return False
    return False