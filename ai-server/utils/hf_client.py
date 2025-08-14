import httpx
from typing import Any, Dict
from ..config import LOCAL_HF_BASE, HF_TIMEOUT

async def _post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{LOCAL_HF_BASE.rstrip('/')}{path}"
    async with httpx.AsyncClient(timeout=HF_TIMEOUT) as client:
        r = await client.post(url, json=payload)
        r.raise_for_status()
        return r.json()

async def call_preprocess(payload: Dict[str, Any]) -> Dict[str, Any]:
    return await _post("/predict_preprocess", payload)

async def call_main(payload: Dict[str, Any]) -> Dict[str, Any]:
    return await _post("/predict_main", payload)

async def call_postprocess(payload: Dict[str, Any]) -> Dict[str, Any]:
    return await _post("/predict_postprocess", payload)