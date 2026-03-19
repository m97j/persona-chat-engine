import json
import os

import torch
import torch.nn as nn
from config import DEVICE, HF_TOKEN
from transformers import AutoModelForCausalLM, AutoTokenizer

SPECIALS = ["<SYS>", "<CTX>", "<PLAYER>", "<NPC>", "<STATE>", "<RAG>", "<PLAYER_STATE>"]

def get_current_branch():
    if os.path.exists("current_branch.txt"):
        with open("current_branch.txt", "r") as f:
            return f.read().strip()
    return "latest"

class ModelWrapper:
    def __init__(self):
        # Flags info
        flags_path = os.path.join(os.path.dirname(__file__), "flags.json")
        self.flags_order = json.load(open(flags_path, encoding="utf-8"))["ALL_FLAGS"]
        self.num_flags = len(self.flags_order)

        branch = get_current_branch()

        # 1) Tokenizer (vocab + SPECIALS at the time of training LoRA)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "m97j/npc_LoRA-fps",
            revision=branch,
            subfolder="testcase_output",
            use_fast=True,
            token=HF_TOKEN,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.tokenizer.add_special_tokens({"additional_special_tokens": SPECIALS})

        # 2) Base model (LoRA model with merged weights, but without custom heads)
        self.model = AutoModelForCausalLM.from_pretrained(
            "m97j/npc_LoRA-fps",
            revision=branch,
            subfolder="testcase_output",
            device_map=None,
            low_cpu_mem_usage=False,
            trust_remote_code=True,
            token=HF_TOKEN
        )

        # 3) add custom heads (delta, flag, flag_threshold) - architecture only, weights will be loaded separately
        hidden_size = self.model.config.hidden_size
        self.model.delta_head = nn.Linear(hidden_size, 2).to(DEVICE)
        self.model.flag_head = nn.Linear(hidden_size, self.num_flags).to(DEVICE)
        self.model.flag_threshold_head = nn.Linear(hidden_size, self.num_flags).to(DEVICE)

        # 4) Load custom head weights separately (if available)
        #  - this is necessary because the LoRA merging process may not include these heads, and they might be trained separately.
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

        # 5) Move model to device and set to eval mode
        self.model.to(DEVICE)
        self.model.eval()

    def get(self):
        return self.tokenizer, self.model, self.flags_order
