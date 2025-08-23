import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

def load_emotion_model(model_name, model_dir):
    if not os.path.exists(model_dir) or not os.listdir(model_dir):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer.save_pretrained(model_dir)
        model.save_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    return tokenizer, model

def load_fallback_model(model_name, model_dir):
    if not os.path.exists(model_dir) or not os.listdir(model_dir):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer.save_pretrained(model_dir)
        model.save_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    return tokenizer, model

def load_embedder(model_name, model_dir):
    if not os.path.exists(model_dir):
        embedder = SentenceTransformer(model_name)
        embedder.save(model_dir)
    embedder = SentenceTransformer(model_dir)
    return embedder