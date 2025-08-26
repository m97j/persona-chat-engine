from fastapi import Request
from models.emotion_model import detect_emotion
from utils.context_parser import ContextParser
from rag.rag_generator import retrieve

def extract_triggers(text: str, parser: ContextParser) -> list:
    stage = parser.game.get("quest_stage", "default")
    trigger_values = parser.npc.get("trigger_values", {}).get(stage, [])
    return [kw for kw in trigger_values if kw in text]

def evaluate_trigger(user_input: str, parser: ContextParser, emotion: dict) -> tuple:
    stage = parser.game.get("quest_stage", "default")
    trigger_def = parser.npc.get("trigger_definitions", {}).get(stage, {})

    required_text = trigger_def.get("required_text", [])
    if required_text and not any(t in user_input for t in required_text):
        return False, trigger_def.get("fallback_style")

    emotion_threshold = trigger_def.get("emotion_threshold", {})
    for emo, threshold in emotion_threshold.items():
        if emotion.get(emo, 0) < threshold:
            return False, trigger_def.get("fallback_style")

    return True, None

async def preprocess_input(
    request: Request,
    session_id: str,
    npc_id: str,
    user_input: str,
    context: dict
) -> dict:
    parser = ContextParser(context)
    emotion = detect_emotion(request, user_input)
    triggers = extract_triggers(user_input, parser)
    is_valid, fb_style_from_rule = evaluate_trigger(user_input, parser, emotion)

    # 공통 필터
    filters = parser.get_filters()
    base_query = f"{parser.npc.get('id', npc_id)}:{filters.get('location','unknown')}:{filters.get('quest_stage','unknown')}"
    main_query = f"{base_query}:main"
    fallback_query = f"{base_query}:fallback"

    rag_main_docs = retrieve(main_query, filters=filters, top_k=5) if is_valid else []
    rag_fallback_docs = retrieve(fallback_query, filters=filters, top_k=5) if not is_valid else []

    fallback_style = fb_style_from_rule or None
    if not is_valid:
        style_dict = {}
        for doc in rag_fallback_docs:
            for line in doc.splitlines():
                key, sep, val = line.partition(":")
                k = key.strip().lower()
                if sep and k in {"style", "npc_action", "npc_emotion"} and val.strip():
                    style_dict[k] = val.strip()
        if style_dict:
            fallback_style = style_dict

    short_history = []
    for h in context.get("dialogue_history", [])[-3:]:
        if "player" in h and "npc" in h:
            short_history.append({"role": "player", "text": h["player"]})
            short_history.append({"role": "npc", "text": h["npc"]})

    return {
        "player_utterance": user_input,
        "npc_id": npc_id,
        "tags": parser.npc,
        "player_state": parser.player,
        "game_state": parser.game,
        "context": short_history,
        "emotion": emotion,
        "triggers": triggers,
        "is_valid": is_valid,
        "fallback_style": fallback_style,
        "rag_main_docs": rag_main_docs,
        "rag_fallback_docs": rag_fallback_docs,
    }
