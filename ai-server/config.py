import os

# Hugging Face Spaces serve URL (공개 설정이므로 직접 주소 사용 가능)
HF_SERVE_URL = os.getenv(
    "HF_SERVE_URL",
    "https://m97j-PersonaChatEngine.hf.space"
)

# 요청 타임아웃 (초 단위)
HF_TIMEOUT = float(os.getenv("HF_TIMEOUT", "25"))

# RAG 항상 사용 (토글이 아니라 고정 사용)
RAG_ENABLED = True

# 생성 파라미터 기본값 (요청마다 override 가능)
GENERATION_CONFIG = {
    "max_new_tokens": int(os.getenv("GEN_MAX_NEW_TOKENS", "220")),
    "temperature": float(os.getenv("GEN_TEMPERATURE", "0.7")),
    "top_p": float(os.getenv("GEN_TOP_P", "0.9")),
    "repetition_penalty": float(os.getenv("GEN_REPETITION_PENALTY", "1.1")),
    "do_sample": True
}

'''
# 모델 정보 (추후 확장 가능)
MODEL_INFO = {
    "base_model": "meta-llama/Meta-Llama-3-8B",
    "adapter": "m97j/PersonaAdapter-v1",
    "serve_mode": "hf_spaces",  # 또는 "local", "api"
}

'''