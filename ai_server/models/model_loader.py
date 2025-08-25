from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM
)
from sentence_transformers import SentenceTransformer


def load_emotion_model(model_name: str, model_dir: Path):
    if not model_dir.exists() or not any(model_dir.iterdir()):
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.save_pretrained(model_dir)
        model.save_pretrained(model_dir)

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir), trust_remote_code=True, local_files_only=True)
    return tokenizer, model


def load_fallback_model(model_name: str, model_dir: Path):
    if not model_dir.exists() or not any(model_dir.iterdir()):
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.save_pretrained(model_dir)
        model.save_pretrained(model_dir)

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), trust_remote_code=True, local_files_only=True)
    return tokenizer, model


def load_embedder(model_name: str, model_dir: Path):
    if not model_dir.exists() or not any(model_dir.iterdir()):
        embedder = SentenceTransformer(model_name)
        embedder.save(str(model_dir))

    embedder = SentenceTransformer(str(model_dir))
    return embedder