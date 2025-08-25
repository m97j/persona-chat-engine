import re
from typing import Tuple, Dict, List, Any
from rag.rag_generator import retrieve

# 1. 모델 응답 검증 및 본문 추출
async def final_check(text: str, user_input: str, context: dict, npc_config: dict) -> dict:
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
    match = re.search(r"<FLAG (.*?) ?/>", text)
    if not match:
        return {}

    raw = match.group(1)
    pairs = re.findall(r'(\w+)="([\d\.]+)"', raw)
    return {k: float(v) for k, v in pairs}

# 3. RAG 기반으로 실제 텍스트 값 추출
def resolve_flag_values(flag_scores: Dict[str, float], npc_id: str, quest_stage: str) -> Dict[str, str]:
    resolved = {}

    for flag_name, score in flag_scores.items():
        query = f"{npc_id}:{quest_stage}:{flag_name}"
        docs = retrieve(query, filters={"npc_id": npc_id})

        for doc in docs:
            metadata = doc.get("metadata", {})
            threshold = float(metadata.get("threshold", 0.8))
            if score >= threshold:
                resolved[flag_name] = metadata.get("value", "")
                break

    return resolved

# 4. 최종 구조화된 결과 반환
async def extract_game_data(text: str, context: dict) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    npc_id = context.get("npc_id", "unknown")
    quest_stage = context.get("quest_stage", "default")

    flag_scores = extract_flag_scores(text)
    resolved_values = resolve_flag_values(flag_scores, npc_id, quest_stage)

    flags = {}
    for key, score in flag_scores.items():
        flags[key] = {
            "score": score,
            "value": resolved_values.get(key)
        }

    # <DELTA mood="..." trust="..." /> 추출
    deltas = []
    delta_match = re.search(r"<DELTA mood=\"(.*?)\" trust=\"(.*?)\" ?/>", text)
    if delta_match:
        deltas.append({
            "mood": delta_match.group(1),
            "trust": float(delta_match.group(2))
        })

    return deltas, flags