import os, json, torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from config import BASE_MODEL, ADAPTER_MODEL, DEVICE

class ModelWrapper:
    def __init__(self):
        # flags.json 로드
        flags_path = os.path.join(os.path.dirname(__file__), "flags.json")
        self.flags_order = json.load(open(flags_path, encoding="utf-8"))["ALL_FLAGS"]
        self.num_flags = len(self.flags_order)

        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(ADAPTER_MODEL, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        # 모델 로드 (최초 빌드/재빌드 시 Hub → 디스크 다운로드, 이후 wake 시 디스크 캐시에서 메모리 로드)
        base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto", trust_remote_code=True)
        self.model = PeftModel.from_pretrained(base, ADAPTER_MODEL, revision="latest", device_map="auto")

        # 커스텀 헤드 부착
        hidden_size = self.model.config.hidden_size
        self.model.delta_head = nn.Linear(hidden_size, 2).to(DEVICE)
        self.model.flag_head = nn.Linear(hidden_size, self.num_flags).to(DEVICE)
        self.model.flag_threshold_head = nn.Linear(hidden_size, self.num_flags).to(DEVICE)

        # 로컬 저장된 헤드 가중치 로드 (있을 경우)
        if os.path.exists("delta_head.pt"):
            self.model.delta_head.load_state_dict(torch.load("delta_head.pt", map_location=DEVICE))
        if os.path.exists("flag_head.pt"):
            self.model.flag_head.load_state_dict(torch.load("flag_head.pt", map_location=DEVICE))
        if os.path.exists("flag_threshold_head.pt"):
            self.model.flag_threshold_head.load_state_dict(torch.load("flag_threshold_head.pt", map_location=DEVICE))

        self.model.eval()

    def get(self):
        return self.tokenizer, self.model, self.flags_order
