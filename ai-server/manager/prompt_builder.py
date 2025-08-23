from rag.rag_generator import retrieve
from typing import Dict
from manager.agent_manager import AgentManager

agent_manager = AgentManager()

def build_main_prompt(preprocessed: dict, context: str, session_id: str, npc_id: str) -> str:
    agent = agent_manager.get_agent(npc_id)

    short_history = preprocessed.get("short_history", [])
    user_input = preprocessed.get("text", "")
    npc_config = preprocessed.get("npc_config", {})
    player_state = {
        "stage": preprocessed.get("quest_stage", "unknown"),
        "location": preprocessed.get("location", "unknown"),
        "relationship": preprocessed.get("relationship", "neutral"),
        "trust": preprocessed.get("trust", "0.5"),
        "reputation": preprocessed.get("player_reputation", "average"),
        "items": preprocessed.get("player_items", []),
        "actions": preprocessed.get("player_actions", []),
        "input": user_input,
    }

    return agent.to_prompt(
        user_input=user_input,
        short_history=short_history,
        context=preprocessed.get("context", {}),
        npc_config=npc_config,
        player_state=player_state
    )

def build_fallback_prompt(npc_config: dict, player_state: dict, emotions: dict, session_id: str, npc_id: str) -> str:
    query = f"{npc_id}:{quest_stage}:trigger"
    filters = {"npc_id": npc_id}
    retrieved_docs = retrieve(query, filters)

    npc_name = npc_config.get("name", "NPC")
    persona = npc_config.get("persona_name", npc_name)
    description = npc_config.get("description", "")
    location = player_state.get("location", "알 수 없음")
    quest_stage = player_state.get("stage", "초기")
    relationship = player_state.get("relationship", "중립")
    trust = player_state.get("trust", "0.5")
    reputation = player_state.get("reputation", "보통")
    style = npc_config.get("style", "기본")
    mood = emotions.get("dominant", "neutral")
    emotion_summary = ", ".join([f"{k}:{round(v,2)}" for k,v in emotions.items()])
    items = player_state.get("items", [])
    actions = player_state.get("actions", [])
    input_text = player_state.get("input", "")

    lore = "\n".join(retrieved_docs)

    return f"""
<FALLBACK>
NPC={npc_name}
Persona={persona}
Description={description}
Location={location}
Quest Stage={quest_stage}
Relationship={relationship}
Trust={trust}
Reputation={reputation}
Mood={mood}
Style={style}
Emotion Summary={emotion_summary}
Items={items}
Actions={actions}
Input="{input_text}"

Lore:
{lore}

Instruction:
플레이어의 입력이 명확하지 않거나 조건을 만족하지 않습니다.
NPC는 현재 설정과 감정 상태, 배경 정보를 기반으로 자연스럽고 몰입감 있는 반응을 생성하십시오.
스토리 진행은 하지 않지만, NPC의 성격에 맞는 대사와 행동을 출력하십시오.
</FALLBACK>
"""