import os
import json
from chromadb import Client
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# 초기화
client = Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory="./chroma_db"))
collection = client.get_or_create_collection(name="game_docs")
embedder = SentenceTransformer("all-MiniLM-L6-v2")


def chroma_initialized() -> bool:
    return os.path.exists("./chroma_db") and len(os.listdir("./chroma_db")) > 0

def load_game_docs_from_disk(path: str) -> list[dict]:
    docs = []
    for filename in os.listdir(path):
        if filename.endswith(".json"):
            with open(os.path.join(path, filename), "r", encoding="utf-8") as f:
                docs.extend(json.load(f))
        elif filename.endswith(".txt"):
            with open(os.path.join(path, filename), "r", encoding="utf-8") as f:
                content = f.read()
                docs.append({
                    "id": filename,
                    "content": content,
                    "metadata": {}
                })
    return docs

# 문서 추가
def add_docs(docs: list[dict]):
    for doc in docs:
        embedding = embedder.encode(doc["content"]).tolist()
        collection.add(
            documents=[doc["content"]],
            embeddings=[embedding],
            metadatas=[doc["metadata"]],
            ids=[doc["id"]]
        )

# 검색
def retrieve(query: str, filters: dict = None, top_k: int = 5) -> list[str]:
    query_embedding = embedder.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=filters
    )
    return results["documents"][0]