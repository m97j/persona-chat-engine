import re
from typing import Tuple, Dict, List, Any
from rag.rag_generator import retrieve

# 1. 모델 응답 검증 및 본문 추출
async def final_check(text: str, user_input: str, context: dict, npc_config: dict) -> dict:
    """
    모델 응답에서 <RESPONSE> 블록을 추출하고 유효성 검증.
    """
    valid = len(text.strip()) > 0 and "에러" not in text and "<RESPONSE>" in text

    response_match = re.search(r"<RESPONSE>(.*?)</RESPONSE>", text, re.DOTALL)
    response_text = response_match.group(1).strip() if response_match else text.strip()

    return {
        "text": response_text,
        "valid": valid,
        "meta": {
            "length": len(response_text),
            "npc": npc_config.get("persona_name", "NPC")
        }
    }

# 2. <FLAG ... /> 태그에서 수치 기반 flag 추출
def extract_flag_scores(text: str) -> Dict[str, float]:
    """
    <FLAG name="..." score="..."/> 형태에서 수치 추출
    """
    match = re.search(r"<FLAG (.*?) ?/>", text)
    if not match:
        return {}

    raw = match.group(1)
    pairs = re.findall(r'(\w+)="([\d\.]+)"', raw)
    return {k: float(v) for k, v in pairs}

# 3. RAG 기반으로 실제 텍스트 값 추출
def resolve_flag_values(flag_scores: Dict[str, float], filters: Dict[str, Any]) -> Dict[str, str]:
    """
    flag_scores를 기반으로 RAG에서 해당 flag의 실제 value를 가져옴.
    filters에는 npc_id, quest_stage, location 등이 포함됨.
    """
    resolved = {}
    npc_id = filters.get("npc_id", "unknown")
    quest_stage = filters.get("quest_stage", "default")

    for flag_name, score in flag_scores.items():
        query = f"{npc_id}:{quest_stage}:{flag_name}"
        docs = retrieve(query, filters=filters)

        for doc in docs:
            metadata = doc.get("metadata", {})
            threshold = float(metadata.get("threshold", 0.8))
            if score >= threshold:
                resolved[flag_name] = metadata.get("value", "")
                break

    return resolved

# 4. 최종 구조화된 결과 반환
async def extract_game_data(text: str, context: dict) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    모델 응답에서 deltas와 flags를 추출.
    context는 preprocess에서 만든 pre 딕셔너리나 최소한 npc_id, game_state, player_state를 포함해야 함.
    """
    npc_id = context.get("npc_id", "unknown")
    game_state = context.get("game_state", {})
    quest_stage = game_state.get("quest_stage", "default")
    location = game_state.get("location", context.get("location", "unknown"))

    filters = {
        "npc_id": npc_id,
        "quest_stage": quest_stage,
        "location": location
    }

    # flag 추출 및 해석
    flag_scores = extract_flag_scores(text)
    resolved_values = resolve_flag_values(flag_scores, filters)

    flags = {}
    for key, score in flag_scores.items():
        flags[key] = {
            "score": score,
            "value": resolved_values.get(key)
        }

    # <DELTA ... /> 추출 (속성 확장 가능)
    deltas = []
    delta_match = re.search(r"<DELTA\s+([^/>]+)\s*/>", text)
    if delta_match:
        attrs = dict(re.findall(r'(\w+)="(.*?)"', delta_match.group(1)))
        # trust 등 숫자는 float 변환 시도
        for k, v in attrs.items():
            try:
                attrs[k] = float(v)
            except ValueError:
                pass
        deltas.append(attrs)

    return deltas, flags
