from typing import List, Dict, Any
from rag.rag_generator import retrieve

class NPCAgent:
    def __init__(self, npc_id: str):
        self.npc_id = npc_id

    def to_prompt(
        self,
        mode: str,
        session_id: str,
        user_input: str,
        short_history: List[dict],
        npc_config: dict,
        player_state: dict,
        game_state: dict,
        fallback_style: dict = None
    ) -> str:
        quest_stage = game_state.get("quest_stage") or npc_config.get("quest_stage", "unknown")
        location = game_state.get("location") or player_state.get("location", "unknown")
        filters = {"npc_id": self.npc_id, "location": location, "quest_stage": quest_stage}
        query = f"{self.npc_id}:{location}:{quest_stage}:{mode}"

        rag_docs = retrieve(query, filters)
        rag_text = "\n".join(f"- {doc}" for doc in rag_docs)

        items = ",".join(player_state.get("items", []))
        actions = ",".join(player_state.get("actions", []))
        ctx_text = "\n".join(f"{t['role']}: {t['text']}" for t in short_history)

        if mode == "main":
            sys_block = f"""<SYS>
NPC_ID={self.npc_id}
SESSION_ID={session_id}
<STATE/>
LOCATION={location}
QUEST_STAGE={quest_stage}
NPC_MOOD={npc_config.get("npc_mood","neutral")}
RELATIONSHIP={npc_config.get("relationship","neutral")}
TRUST={npc_config.get("trust","0.5")}
PLAYER_REPUTATION={player_state.get("reputation","average")}
STYLE={npc_config.get("style","neutral")}
ITEMS={items}
ACTIONS={actions}
FORMAT:
  <RESPONSE>...</RESPONSE>
  <DELTA trust="..." relationship="..." />
  <FLAG name="..." />
</SYS>"""
            rag_block = f"<RAG>\n{rag_text}\n</RAG>" if rag_text else "<RAG/>"
            ctx_block = f"<CTX>\n{ctx_text}\n</CTX>" if ctx_text else "<CTX/>"
            player_block = f"<PLAYER>{user_input}</PLAYER>\n<NPC>"
            return f"{sys_block}\n{rag_block}\n{ctx_block}\n{player_block}"

        elif mode == "fallback":
            instr = "조건 불충족. 스토리 진행은 하지 않고, 캐릭터 일관성을 유지하며 자연스럽게 응답하라."
            if fallback_style:
                s = fallback_style.get("style"); a = fallback_style.get("npc_action"); e = fallback_style.get("npc_emotion")
                more = []
                if s: more.append(f"대화 스타일={s}")
                if a: more.append(f"NPC 행동={a}")
                if e: more.append(f"NPC 감정={e}")
                if more:
                    instr += " " + "; ".join(more) + "."

            return f"""
<FALLBACK>
NPC_ID={self.npc_id}
SESSION_ID={session_id}
LOCATION={location}
QUEST_STAGE={quest_stage}
MOOD={npc_config.get("npc_mood","neutral")}
STYLE={npc_config.get("style","neutral")}
ITEMS={items}
ACTIONS={actions}
EMOTION_SUMMARY={', '.join([f"{k}:{round(v,2)}" for k,v in npc_config.get('emotion',{}).items()])}
INPUT="{user_input}"

RAG:
{rag_text or "(none)"}

INSTRUCTION:
{instr}
</FALLBACK>
""".strip()

class AgentManager:
    def __init__(self):
        self.agents: Dict[str, NPCAgent] = {}

    def get_agent(self, npc_id: str) -> NPCAgent:
        if npc_id not in self.agents:
            self.agents[npc_id] = NPCAgent(npc_id)
        return self.agents[npc_id]

# 전역 인스턴스
agent_manager = AgentManager()
