# 간단 Chroma + SentenceTransformers 기반 RAG
import os
from typing import List, Dict, Any
from chromadb.config import Settings
import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
_embed_model = SentenceTransformer("all-MiniLM-L6-v2")

_client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=CHROMA_DIR))
_collection = _client.get_or_create_collection(name="game_knowledge")

def add_docs(docs: List[Dict[str, Any]]):
    for d in docs:
        text = d["text"]
        emb = _embed_model.encode(text).tolist()
        _collection.upsert(
            ids=[d["id"]],
            metadatas=[d.get("meta", {})],
            documents=[text],
            embeddings=[emb]
        )
    _client.persist()

def retrieve(query: str, k: int = 4) -> List[Dict[str, Any]]:
    q_emb = _embed_model.encode(query).tolist()
    results = _collection.query(query_embeddings=[q_emb], n_results=k)
    docs = []
    if results["documents"]:
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            docs.append({"text": doc, "meta": meta})
    return docs