import torch

# 모델 경로
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_MODEL = "m97j/npc-LoRA-fps"

# 장치 설정
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 토크나이저/모델 공통
MAX_LENGTH = 1024
NUM_FLAGS = 7  # flags.json 길이와 일치

# 생성 파라미터
GEN_MAX_NEW_TOKENS = 200
GEN_TEMPERATURE = 0.7
GEN_TOP_P = 0.9
