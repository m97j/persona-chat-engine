from fastapi import Request
from pipeline.preprocess import preprocess_input
from pipeline.generator import generate_response
from pipeline.postprocess import final_check, extract_game_data
from models.fallback_model import generate_fallback_response
from manager.prompt_builder import build_main_prompt, build_fallback_prompt

async def handle_dialogue(
    request: Request,
    session_id: str,
    npc_id: str,
    user_input: str,
    context: dict,
    npc_config: dict
) -> dict:
    # 1. 입력 전처리
    pre = await preprocess_input(request, session_id, npc_id, user_input, context)

    # 2. 유효하지 않은 입력일 경우 fallback 처리
    if not pre.get("is_valid", True):
        fallback_prompt = build_fallback_prompt(
            npc_config=npc_config,
            player_state=pre.get("player_state", {}),
            emotion=pre.get("emotion", ""),
            session_id=session_id,
            npc_id=npc_id
        )
        fallback_text = await generate_fallback_response(request, fallback_prompt)
        post = await final_check(fallback_text, user_input, context, npc_config)

        return {
            "npc_output_text": post["text"],
            "flags": {"trigger": "fallback"},
            "deltas": [],
            "valid": post["valid"],
            "meta": post["meta"]
        }

    # 3. 메인 프롬프트 생성 및 응답 생성
    prompt = build_main_prompt(
        pre=pre,
        history=context.get("history", ""),
        session_id=session_id,
        npc_id=npc_id
    )
    generated_text = await generate_response(session_id, npc_id, prompt)

    # 4. 응답 후처리 및 flag/delta 추출
    post = await final_check(generated_text, user_input, context, npc_config)
    deltas, flags = await extract_game_data(post["text"], context)

    # 5. 최종 결과 반환
    return {
        "npc_output_text": post["text"],
        "flags": flags,
        "deltas": deltas,
        "valid": post["valid"],
        "meta": post["meta"]
    }