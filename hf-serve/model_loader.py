import os, json, torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from config import BASE_MODEL, ADAPTERS, DEVICE, HF_TOKEN

ADAPTER_VOCAB_SIZE = 151672  # 학습 시점 vocab size (로그 기준)

SPECIALS = ["<SYS>", "<CTX>", "<PLAYER>", "<NPC>", "<STATE>", "<RAG>", "<PLAYER_STATE>"]

def get_current_branch():
    if os.path.exists("current_branch.txt"):
        with open("current_branch.txt", "r") as f:
            return f.read().strip()
    return "latest"

class ModelWrapper:
    def __init__(self):
        # Flags 정보
        flags_path = os.path.join(os.path.dirname(__file__), "flags.json")
        self.flags_order = json.load(open(flags_path, encoding="utf-8"))["ALL_FLAGS"]
        self.num_flags = len(self.flags_order)

        # 1) 토크나이저 (학습과 동일 옵션 + SPECIALS)
        self.tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL,
            use_fast=True,
            token=HF_TOKEN,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        # 학습 시 추가했던 특수 토큰 재현
        self.tokenizer.add_special_tokens({"additional_special_tokens": SPECIALS})

        # 2) 베이스 모델 (오프로딩 끄고 로드)
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            device_map=None,              # ✅ 오프로딩 비활성화
            low_cpu_mem_usage=False,      # ✅ meta 텐서 생성 방지
            trust_remote_code=True,
            token=HF_TOKEN
        )

        # 3) 학습 시 vocab size로 강제 리사이즈 (어댑터 로드 전에)
        base.resize_token_embeddings(ADAPTER_VOCAB_SIZE)

        # 4) LoRA 어댑터 적용 (오프로딩 끄고 로드)
        branch = get_current_branch()
        self.model = PeftModel.from_pretrained(
            base,
            ADAPTERS,
            revision=branch,
            device_map=None,              # ✅ 오프로딩 비활성화
            low_cpu_mem_usage=False,      # ✅ meta 텐서 생성 방지
            token=HF_TOKEN
        )

        # 5) 커스텀 헤드
        hidden_size = self.model.config.hidden_size
        self.model.delta_head = nn.Linear(hidden_size, 2).to(DEVICE)
        self.model.flag_head = nn.Linear(hidden_size, self.num_flags).to(DEVICE)
        self.model.flag_threshold_head = nn.Linear(hidden_size, self.num_flags).to(DEVICE)

        # 6) 커스텀 헤드 가중치 로드(있을 경우)
        for head_name, file_name in [
            ("delta_head", "delta_head.pt"),
            ("flag_head", "flag_head.pt"),
            ("flag_threshold_head", "flag_threshold_head.pt")
        ]:
            try:
                if os.path.exists(file_name):
                    getattr(self.model, head_name).load_state_dict(
                        torch.load(file_name, map_location=DEVICE)
                    )
            except Exception as e:
                print(f"[WARN] Failed to load {file_name}: {e}")

        # 7) 디바이스 배치
        self.model.to(DEVICE)
        self.model.eval()

    def get(self):
        return self.tokenizer, self.model, self.flags_order
