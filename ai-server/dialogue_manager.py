import os
from typing import Dict, Any, List
from .preprocess import preprocess
from .postprocess import validate_and_fix
from .generator import generate
from .agent_manager import NPCAgent
from .config import RAG_ENABLED

if RAG_ENABLED:
    from .rag import retrieve
else:
    retrieve = None  # type: ignore

async def handle_interaction(
    session_id: str,
    npc_id: str,
    user_input: str,
    context: Dict[str, Any],
    short_history: List[Dict[str, str]],
    persona_tag: str | None,
    gen_params: Dict[str, Any] | None,
):
    pre = await preprocess(session_id, npc_id, user_input, context)
    flags: Dict[str, Any] = {}
    meta: Dict[str, Any] = {}

    # 게이트 차단 → main 생략 + policy_action 후처리
    if pre.get("gate") == "deny":
        action = pre.get("action", "ignore")
        text, valid, post_meta = await validate_and_fix(
            session_id, npc_id, generated_text="", constraints=(context or {}).get("constraints", {}),
            policy_action=action
        )
        flags.update({"action": action, "trigger": pre.get("trigger")})
        return {"npc_response": text, "flags": flags, "valid": valid, "meta": post_meta}

    # 허용 → prompt 구성 → main → postprocess
    agent = NPCAgent(npc_id, persona_tag=persona_tag)
    docs = retrieve(user_input, k=4) if RAG_ENABLED and retrieve else None
    prompt = agent.to_prompt(user_input, short_history or [], retrieved_docs=docs, context=context)

    gen_text, hf_meta = await generate(session_id, npc_id, prompt, gen_params)
    post_text, valid, post_meta = await validate_and_fix(
        session_id, npc_id, gen_text, (context or {}).get("constraints", {})
    )
    meta = {"hf": hf_meta, "post": post_meta}
    flags["trigger"] = pre.get("trigger")
    flags["last_utterance"] = bool(post_meta.get("last_utterance", False))
    return {"npc_response": post_text, "flags": flags, "valid": valid, "meta": meta}