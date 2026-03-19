import os

import torch
from dotenv import load_dotenv

# Load .env file (for local development)
load_dotenv()

# Model path (uses default if environment variable is missing)
BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
ADAPTERS = os.getenv("ADAPTER_MODEL", "m97j/npc_LoRA-fps")

# Device configuration
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

# Tokenizer/Model common parameters
MAX_LENGTH = int(os.getenv("MAX_LENGTH", 1024))
NUM_FLAGS = int(os.getenv("NUM_FLAGS", 7))  # match withflags.json

# Generation parameters (can be overridden at inference time)
GEN_MAX_NEW_TOKENS = int(os.getenv("GEN_MAX_NEW_TOKENS", 400))
GEN_TEMPERATURE = float(os.getenv("GEN_TEMPERATURE", 0.7))
GEN_TOP_P = float(os.getenv("GEN_TOP_P", 0.9))

# Hugging Face Token (For Private Model Access)
HF_TOKEN = os.getenv("HF_TOKEN")
