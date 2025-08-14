import os

# HF-serve base URL (e.g., http://hf-serve:8000)
LOCAL_HF_BASE = os.getenv("LOCAL_HF_URL", "http://hf-serve:8000")

# Timeouts
HF_TIMEOUT = float(os.getenv("HF_TIMEOUT", "25"))

# RAG toggle
RAG_ENABLED = os.getenv("RAG_ENABLED", "0") == "1"

# Generation defaults (can be overridden per request)
GEN_DEFAULTS = {
    "max_new_tokens": int(os.getenv("GEN_MAX_NEW_TOKENS", "220")),
    "temperature": float(os.getenv("GEN_TEMPERATURE", "0.7")),
    "top_p": float(os.getenv("GEN_TOP_P", "0.9")),
}