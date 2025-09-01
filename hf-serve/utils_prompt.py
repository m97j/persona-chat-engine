from typing import Dict, Any

def build_webtest_prompt(npc_id: str, npc_location: str, player_utt: str) -> str:
    # 웹 테스트에서는 최소 필드만 채운 pre dict 생성
    pre = {
        "tags": {
            "npc_id": npc_id,
            "location": npc_location,
            "quest_stage": "",
            "relationship": "",
            "trust": "",
            "npc_mood": "",
            "player_reputation": "",
            "style": ""
        },
        "player_state": {},
        "rag_main_docs": [],  # 웹 테스트에서는 RAG 문서 없음
        "context": [],        # 대화 히스토리 없음
        "player_utterance": player_utt
    }
    # session_id는 웹 테스트에서는 의미 없으니 빈 값
    return build_main_prompt(pre, session_id="", npc_id=npc_id)


def build_main_prompt(pre: Dict[str, Any], session_id: str, npc_id: str) -> str:
    tags = pre.get("tags", {})
    ps = pre.get("player_state", {})
    rag_docs = pre.get("rag_main_docs", [])

    # RAG 문서 분리
    lore_text = ""
    desc_text = ""
    for doc in rag_docs:
        if "LORE:" in doc:
            lore_text += doc + "\n"
        elif "DESCRIPTION:" in doc:
            desc_text += doc + "\n"
        else:
            # fallback: type 기반 분리 가능
            if "lore" in doc.lower():
                lore_text += doc + "\n"
            elif "description" in doc.lower():
                desc_text += doc + "\n"

    prompt = [
        "<SYS>",
        f"NPC_ID={tags.get('npc_id','')}",
        f"NPC_LOCATION={tags.get('location','')}",
        "TAGS:",
        f" quest_stage={tags.get('quest_stage','')}",
        f" relationship={tags.get('relationship','')}",
        f" trust={tags.get('trust','')}",
        f" npc_mood={tags.get('npc_mood','')}",
        f" player_reputation={tags.get('player_reputation','')}",
        f" style={tags.get('style','')}",
        "</SYS>",
        "<RAG>",
        f"LORE: {lore_text.strip() or '(없음)'}",
        f"DESCRIPTION: {desc_text.strip() or '(없음)'}",
        "</RAG>",
        "<PLAYER_STATE>"
    ]

    if ps.get("items"):
        prompt.append(f"items={','.join(ps['items'])}")
    if ps.get("actions"):
        prompt.append(f"actions={','.join(ps['actions'])}")
    if ps.get("position"):
        prompt.append(f"position={ps['position']}")
    prompt.append("</PLAYER_STATE>")

    prompt.append("<CTX>")
    for h in pre.get("context", []):
        prompt.append(f"{h['role']}: {h['text']}")
    prompt.append("</CTX>")

    prompt.append(f"<PLAYER>{pre.get('player_utterance','').rstrip()}")
    prompt.append("<STATE>")
    prompt.append("<NPC>")

    return "\n".join(prompt)