import os
from typing import Dict, Any

BASE_DIR = os.getenv("MODEL_BASE", "/models")  # mount point

class ModelManager:
    def __init__(self):
        self.paths = {
            "preprocess": os.path.join(BASE_DIR, "preprocess_model"),
            "main": os.path.join(BASE_DIR, "main_dialogue_model"),
            "postprocess": os.path.join(BASE_DIR, "postprocess_model"),
        }
        # 실제 서비스에서는 여기서 tokenizer/model/peft adapter 로드 or lazy-load
        self.models = {}

    def has_model(self, kind: str) -> bool:
        p = self.paths.get(kind)
        return bool(p and os.path.exists(p) and any(os.listdir(p)))

    # ===== Stub implementations (데모용) =====
    def predict_preprocess(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        text = payload.get("text", "")
        # 간단 감정/트리거 시연
        if any(k in text for k in ["아이", "죽", "사망"]):
            return {"gate": "allow", "trigger": "grief", "source": "local-model"}
        # 게이트 규칙은 ai-server에서도 하므로 여기서는 allow 기본
        return {"gate": "allow", "source": "local-model"}

    def predict_main(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        prompt = payload.get("prompt", "")
        gen = payload.get("gen", {})
        # 실제 구현: HF 모델 generate 호출 (LoRA/RoRA 어댑터 포함)
        # return {"text": generated_text, "meta": {...}}
        return {
            "text": f"(LOCAL MODEL) {prompt[:300]}",
            "meta": {"echo": True, "used_gen_params": gen}
        }

    def predict_postprocess(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        text = payload.get("text", "")
        constraints = payload.get("constraints", {})
        cleaned = False
        if constraints.get("no_violence") and "죽여" in text:
            text = text.replace("죽여", "[redacted]")
            cleaned = True
        # 마지막 대사 매우 단순 판정(실서비스는 모델/룰 강화)
        last = any(k in text for k in ["작별", "안녕히", "다음에 보자", "그만하자", "여기까지"])
        return {
            "text": text,
            "valid": True,
            "meta": {"cleaned": cleaned, "last_utterance": last}
        }