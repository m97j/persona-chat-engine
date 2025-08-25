from fastapi import Request
from models.emotion_model import detect_emotion
from utils.context_parser import ContextParser

def preprocess_input(request: Request, context: dict, text: str) -> dict:
    parser = ContextParser(context)

    emotion = detect_emotion(request, text)
    triggers = extract_triggers(text, parser)

    is_valid, fallback_style = evaluate_trigger(text, parser, emotion)

    return {
        "text": text,
        "emotion": emotion,
        "triggers": triggers,
        "player_state": parser.player,
        "npc_config": parser.npc,
        "game_state": parser.game,
        "is_valid": is_valid,
        "fallback_style": fallback_style
    }

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