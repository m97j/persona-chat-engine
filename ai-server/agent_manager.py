from typing import Optional, List, Dict, Any

class NPCAgent:
    """
    단순 Persona Tag 기반 프롬프트 빌더.
    - LoRA/RoRA 어댑터 학습 시 사용한 태그 규약: [NPC=<npc_id>] 혹은 요청에서 전달된 persona_tag
    """
    def __init__(self, npc_id: str, persona_tag: Optional[str] = None):
        self.npc_id = npc_id
        self.persona_tag = persona_tag or f"[NPC={npc_id}]"

    def to_prompt(
        self,
        user_input: str,
        short_history: List[Dict[str, str]],
        retrieved_docs: Optional[List[Dict[str, Any]]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        lines = [self.persona_tag]

        if context:
            # 과도한 노출 방지: 핵심 상태만 요약
            essentials = {
                "quest_flags": context.get("quest_flags"),
                "reputation": context.get("reputation"),
            }
            lines.append(f"Context: {essentials}")

        if retrieved_docs:
            lines.append("Knowledge:")
            for d in retrieved_docs:
                lines.append(f"- {d.get('text','')}")

        for h in short_history[-8:]:
            role = "Player" if h.get("role") == "user" else "NPC"
            lines.append(f"{role}: {h.get('text','')}")

        lines.append(f"Player: {user_input}")
        lines.append("NPC:")
        return "\n".join(lines)