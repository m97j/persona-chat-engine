from typing import Any, Dict


def build_webtest_prompt(npc_id: str, npc_location: str, player_utt: str) -> str:
    """
    Web Test Only: Generate a prompt string suitable for the model training format 
    using minimum input values (NPC ID, Location, Player utterance).
    """
    pre = {
        "npc_id": npc_id,
        "npc_location": npc_location,
        "tags": {
            "quest_stage": "",
            "relationship": "",
            "trust": "",
            "npc_mood": "",
            "player_reputation": "",
            "style": ""
        },
        "player_state": {
            "items": [],
            "actions": [],
            "position": ""
        },
        "rag_main_docs": [],
        "context": [],
        "player_utterance": player_utt
    }
    return _assemble_prompt_for_model(pre)

def _assemble_prompt_for_model(pre: Dict[str, Any]) -> str:
    """
    Web Test Only: Internal function for assembling the prompt string for the model.
    pre dict → Model input format string (<SYS>~<NPC>)
    """

    tags = pre.get("tags", {})
    ps = pre.get("player_state", {})
    rag_docs = pre.get("rag_main_docs", [])

    # RAG documents are categorized into LORE and DESCRIPTION based on their content.
    lore_text = ""
    desc_text = ""
    for doc in rag_docs:
        if "LORE:" in doc:
            lore_text += doc + "\n"
        elif "DESCRIPTION:" in doc:
            desc_text += doc + "\n"
        else:
            if "lore" in doc.lower():
                lore_text += doc + "\n"
            elif "description" in doc.lower():
                desc_text += doc + "\n"

    prompt = [
        "<SYS>",
        f"NPC_ID={pre.get('npc_id','')}",
        f"NPC_LOCATION={pre.get('npc_location','')}",
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
        "<PLAYER_STATE>",
        f"items={','.join(ps.get('items', []))}" if ps.get("items") else "items=",
        f"actions={','.join(ps.get('actions', []))}" if ps.get("actions") else "actions=",
        f"position={ps.get('position','')}",
        "</PLAYER_STATE>",
        "<CTX>"
    ]

    for h in pre.get("context", []):
        prompt.append(f"{h['role']}: {h['text']}")
    prompt.append("</CTX>")

    prompt.append(f"<PLAYER>{pre.get('player_utterance','').rstrip()}")
    prompt.append("<STATE>")
    prompt.append("<NPC>")

    return "\n".join(prompt)
