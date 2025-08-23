from typing import Optional, List, Dict, Any
from rag.rag_generator import retrieve

class NPCAgent:
    """
    Persona 기반 프롬프트 빌더.
    - NPC ID와 태그 기반으로 프롬프트를 구성
    - 컨텍스트, 히스토리, RAG 문서, 상태 정보를 활용
    - 메인 모델 학습 프롬프트 형식에 맞춰 출력
    """
    def __init__(self, npc_id: str, persona_tag: Optional[str] = None):
        self.npc_id = npc_id
        self.persona_tag = persona_tag or f"[NPC={npc_id}]"

    def to_prompt(
        self,
        user_input: str,
        short_history: List[Dict[str, str]],
        context: Optional[Dict[str, Any]] = None,
        npc_config: Optional[Dict[str, Any]] = None,
        player_state: Optional[Dict[str, Any]] = None,
    ) -> str:
        # 기본 상태값
        quest_stage = player_state.get("stage", "unknown") if player_state else "unknown"
        location = player_state.get("location", "unknown")
        relationship = player_state.get("relationship", "neutral")
        trust = player_state.get("trust", "0.5")
        reputation = player_state.get("reputation", "average")
        style = player_state.get("style", "neutral")
        items = ",".join(player_state.get("items", []))
        actions = ",".join(player_state.get("actions", []))
        trigger_matched = npc_config.get("trigger_matched", "")
        action = npc_config.get("action", "none")
        flags = npc_config.get("flags", "none")

        # RAG 문서
        query = f"{self.npc_id}:{quest_stage}:trigger"
        filters = {"npc_id": self.npc_id}
        retrieved_docs = retrieve(query, filters)
        lore_text = "\n".join(f"- {doc.get('description', '')}" for doc in retrieved_docs)

        # 컨텍스트
        context_text = context.get("context", "") if context else ""

        # <SYS> 블록
        sys_block = f"""<SYS>
NPC_ID={self.npc_id}
QUEST_STAGE={quest_stage}
PLAYER_STATE:
  location={location}
  relationship={relationship}
  trust={trust}
  npc_mood={npc_config.get("mood", "neutral")}
  reputation={reputation}
  style={style}
  items={items}
  actions={actions}
  input="{user_input}"
TRIGGER_MATCHED={trigger_matched}
ACTION={action}
FLAGS={flags}
LORE:
{lore_text}
FORMAT:
  <RESPONSE>...</RESPONSE>
  <DELTA mood="{npc_config.get("mood", "neutral")}" trust="{trust}" />
  <FLAG relevance="high" />
</SYS>"""

        # <CTX> 블록
        ctx_block = f"<CTX>\n{context_text.strip()}\n</CTX>"

        # 히스토리 블록 (선택적으로 포함 가능)
        history_block = ""
        for h in short_history[-8:]:
            role = "Player" if h.get("role") == "user" else "NPC"
            history_block += f"{role}: {h.get('text', '')}\n"

        # <PLAYER> 블록
        player_block = f"<PLAYER>{user_input}</PLAYER>\n<NPC>"

        return f"{sys_block}\n{ctx_block}\n{history_block}{player_block}"
    

class AgentManager:
    """
    NPC별 Agent 인스턴스를 관리하는 매니저
    """
    def __init__(self):
        self.agents: Dict[str, NPCAgent] = {}

    def get_agent(self, npc_id: str) -> NPCAgent:
        if npc_id not in self.agents:
            self.agents[npc_id] = NPCAgent(npc_id)
        return self.agents[npc_id]