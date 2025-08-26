from fastapi import Request
from pipeline.preprocess import preprocess_input
from pipeline.generator import generate_response
from pipeline.postprocess import final_check, extract_game_data
from models.fallback_model import generate_fallback_response
from manager.agent_manager import agent_manager

async def handle_dialogue(
    request: Request,
    session_id: str,
    npc_id: str,
    user_input: str,
    context: dict,
    npc_config: dict
) -> dict:
    pre = await preprocess_input(request, session_id, npc_id, user_input, context)
    agent = agent_manager.get_agent(npc_id)

    # --- Fallback 경로 ---
    if not pre.get("is_valid", True):
        fb_prompt = agent.to_prompt(
            mode="fallback",
            session_id=session_id,
            user_input=pre["player_utterance"],
            short_history=pre.get("context", []),
            npc_config=pre["tags"],
            player_state=pre["player_state"],
            game_state=pre["game_state"],
            fallback_style=pre.get("fallback_style")
        )
        fb_raw = await generate_fallback_response(request, fb_prompt)
        post = await final_check(fb_raw, user_input, context, npc_config)
        return {
            "npc_output_text": post["text"],
            "flags": {"trigger": "fallback"},
            "deltas": [],
            "valid": post["valid"],
            "meta": post["meta"]
        }

    # --- Main 경로 ---
    main_prompt = agent.to_prompt(
        mode="main",
        session_id=session_id,
        user_input=pre["player_utterance"],
        short_history=pre.get("context", []),
        npc_config=pre["tags"],
        player_state=pre["player_state"],
        game_state=pre["game_state"]
    )
    result = await generate_response(session_id, npc_id, main_prompt, max_tokens=200)
    generated_text = result.get("text", "...")

    post = await final_check(generated_text, user_input, context, npc_config)
    deltas, flags = await extract_game_data(post["text"], context)

    # 모델이 직접 준 delta/flag도 병합
    model_deltas = result.get("delta") or []
    model_flags = result.get("flag") or []

    return {
        "npc_output_text": post["text"],
        "flags": flags or model_flags,
        "deltas": deltas or model_deltas,
        "valid": post["valid"],
        "meta": post["meta"]
    }
