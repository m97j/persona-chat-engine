from typing import Dict, Any, Optional, Tuple
from .utils.model_loader import hf_has_model
from .utils.hf_client import call_postprocess

def _local_ethics_and_last(text: str) -> Tuple[str, bool, Dict[str, Any]]:
    # 마지막 대사 추정(아주 단순 규칙)
    last = any(k in text for k in ["작별", "안녕히", "다음에 보자", "그만하자", "여기까지"])
    cleaned = False
    if "죽여" in text:
        text = text.replace("죽여", "[redacted]")
        cleaned = True
    return text, True, {"cleaned": cleaned, "last_utterance": last}

def _policy_action_reply(action: str) -> str:
    tone_map = {
        "ignore":  "NPC는 당신을 힐끗 보더니 대화를 이어가지 않는다.",
        "expel":   "여긴 네가 있을 곳이 아니다. 당장 나가.",
        "attack":  "NPC가 무기를 뽑으며 위협한다!",
        "flee":    "NPC가 겁먹은 듯 뒤로 물러나더니 달아난다."
    }
    return tone_map.get(action, "NPC는 반응하지 않는다.")

async def validate_and_fix(
    session_id: str,
    npc_id: str,
    generated_text: str,
    constraints: Dict[str, Any],
    *,
    policy_action: Optional[str] = None,
):
    # main 생략 케이스: 정책행동 응답
    if policy_action:
        text = _policy_action_reply(policy_action)
        text, valid, meta = _local_ethics_and_last(text)
        meta["policy_action"] = policy_action
        return text, valid, meta

    # 로컬 postprocess 모델 사용(있으면)
    if await hf_has_model("postprocess"):
        try:
            resp = await call_postprocess({
                "session_id": session_id,
                "npc_id": npc_id,
                "text": generated_text,
                "constraints": constraints or {}
            })
            # resp: {text, valid, meta}
            return resp.get("text", generated_text), resp.get("valid", True), resp.get("meta", {})
        except Exception:
            pass

    # 간이 윤리/마지막 대사 판정
    return _local_ethics_and_last(generated_text)