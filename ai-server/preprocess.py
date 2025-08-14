from typing import Dict, Any
from .utils.model_loader import hf_has_model
from .utils.hf_client import call_preprocess

# 규칙 게이트(예시): 필수 아이템, 쿨다운, 폭력 표현 차단 등
def _rule_gate(context: Dict[str, Any] | None, user_input: str) -> Dict[str, Any]:
    ctx = context or {}
    inv = set(ctx.get("inventory", []))
    qf  = ctx.get("quest_flags", {})
    cd  = ctx.get("cooldowns", {})

    # 예시1: 무기 강화 관련 대화는 금전 필요
    if any(k in user_input for k in ["강화", "날카롭게", "연마"]) and "coin_pouch" not in inv:
        return {"gate": "deny", "action": "ignore", "reason": "need_coins"}

    # 예시2: 장소 쿨다운
    if cd.get("blacksmith_shop", 0) > 0:
        return {"gate": "deny", "action": "expel", "reason": "cooldown"}

    # 예시3: 폭력/금칙어
    if any(t in user_input for t in ["죽여", "불태워", "훔쳐"]):
        return {"gate": "deny", "action": "expel", "reason": "violent_request"}

    return {"gate": "allow"}

async def preprocess(session_id: str, npc_id: str, text: str, context: Dict[str, Any]):
    # 1) 로컬 전처리 모델이 있으면 우선 사용
    if await hf_has_model("preprocess"):
        try:
            return await call_preprocess({
                "session_id": session_id,
                "npc_id": npc_id,
                "text": text,
                "context": context or {}
            })
        except Exception:
            # 모델 실패 시 규칙 게이트로 폴백
            pass

    # 2) 규칙 게이트
    return _rule_gate(context, text)