from typing import Dict, List

def _format_history(ctx: List[dict]) -> str:
    return "\n".join(f"{t['role']}: {t['text']}" for t in ctx)

def build_main_prompt(pre: dict, session_id: str, npc_id: str, max_history: int = 6) -> str:
    tags = pre.get("tags", {})
    ps = pre.get("player_state", {})
    gs = pre.get("game_state", {})
    rag_text = "\n".join(f"- {doc}" for doc in pre.get("rag_main_docs", []))
    ctx_text = _format_history(pre.get("context", [])[-max_history:])

    items = ",".join(ps.get("items", []))
    actions = ",".join(ps.get("actions", []))
    location = gs.get("location") or ps.get("location", "unknown")
    quest_stage = gs.get("quest_stage", "unknown")

    sys_block = f"""<SYS>
NPC_ID={npc_id}
SESSION_ID={session_id}
<STATE/>
LOCATION={location}
QUEST_STAGE={quest_stage}
NPC_MOOD={tags.get("npc_mood","neutral")}
RELATIONSHIP={tags.get("relationship","neutral")}
TRUST={tags.get("trust","0.5")}
PLAYER_REPUTATION={ps.get("reputation","average")}
STYLE={tags.get("style","neutral")}
ITEMS={items}
ACTIONS={actions}
FORMAT:
  <RESPONSE>...</RESPONSE>
  <DELTA trust="..." relationship="..." />
  <FLAG name="..." />
</SYS>"""

    rag_block = f"<RAG>\n{rag_text}\n</RAG>" if rag_text else "<RAG/>"
    ctx_block = f"<CTX>\n{ctx_text}\n</CTX>" if ctx_text else "<CTX/>"
    player_block = f"<PLAYER>{pre['player_utterance']}</PLAYER>\n<NPC>"

    return f"{sys_block}\n{rag_block}\n{ctx_block}\n{player_block}"

def build_fallback_prompt(pre: dict, session_id: str, npc_id: str) -> str:
    tags = pre.get("tags", {})
    ps = pre.get("player_state", {})
    gs = pre.get("game_state", {})
    rag_text = "\n".join(f"- {doc}" for doc in pre.get("rag_fallback_docs", []))
    fb = pre.get("fallback_style") or {}

    items = ",".join(ps.get("items", []))
    actions = ",".join(ps.get("actions", []))
    location = gs.get("location") or ps.get("location", "unknown")
    quest_stage = gs.get("quest_stage", "unknown")

    instr = "조건 불충족. 스토리 진행은 하지 않고, 캐릭터 일관성을 유지하며 자연스럽게 응답하라."
    if fb:
        # 선택적 구체화
        s = fb.get("style"); a = fb.get("npc_action"); e = fb.get("npc_emotion")
        more = []
        if s: more.append(f"대화 스타일={s}")
        if a: more.append(f"NPC 행동={a}")
        if e: more.append(f"NPC 감정={e}")
        if more:
            instr += " " + "; ".join(more) + "."

    return f"""
<FALLBACK>
NPC_ID={npc_id}
SESSION_ID={session_id}
LOCATION={location}
QUEST_STAGE={quest_stage}
MOOD={tags.get("npc_mood","neutral")}
STYLE={tags.get("style","neutral")}
ITEMS={items}
ACTIONS={actions}
EMOTION_SUMMARY={', '.join([f"{k}:{round(v,2)}" for k,v in pre.get('emotion',{}).items()])}
INPUT="{pre['player_utterance']}"

RAG:
{rag_text or "(none)"}

INSTRUCTION:
{instr}
</FALLBACK>
""".strip()
