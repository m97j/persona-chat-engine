from typing import Dict, Any

def build_webtest_prompt(npc_id: str, npc_location: str, player_utt: str) -> str:
    """
    Web Test 전용: 최소 입력값(NPC ID, Location, Player 발화)으로
    모델 학습 포맷에 맞는 prompt 문자열을 생성.
    실제 API/게임 서비스 경로에서는 사용하지 않음.
    """
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
        "rag_main_docs": [],
        "context": [],
        "player_utterance": player_utt
    }
    return _assemble_prompt_for_model(pre)

def _assemble_prompt_for_model(pre: Dict[str, Any]) -> str:
    """
    Web Test 전용 내부 함수:
    pre dict → 모델 입력 포맷 문자열(<SYS>~<NPC>)
    """

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