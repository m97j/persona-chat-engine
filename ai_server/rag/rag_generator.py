import os, json
from typing import List, Dict, Any, Optional
from chromadb import PersistentClient

_client = PersistentClient(path="./chroma_db")
_collection = _client.get_or_create_collection(name="game_docs")
_embedder = None

def set_embedder(embedder):
    global _embedder
    _embedder = embedder

def chroma_initialized() -> bool:
    return os.path.exists("./chroma_db") and len(os.listdir("./chroma_db")) > 0

def load_game_docs_from_disk(path: str) -> List[Dict[str, Any]]:
    docs = []
    for filename in os.listdir(path):
        full = os.path.join(path, filename)
        if filename.endswith(".json"):
            with open(full, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    docs.extend(data)
                else:
                    docs.append(data)
        elif filename.endswith(".txt"):
            with open(full, "r", encoding="utf-8") as f:
                content = f.read()
                docs.append({"id": filename, "content": content, "metadata": {}})
    return docs

def add_docs(docs: List[Dict[str, Any]]):
    assert _embedder is not None, "Embedder not initialized"
    for doc in docs:
        assert "id" in doc and "content" in doc, "doc requires id and content"
        meta = doc.get("metadata", {})
        emb = _embedder.encode(doc["content"]).tolist()
        _collection.add(
            documents=[doc["content"]],
            embeddings=[emb],
            metadatas=[meta],
            ids=[doc["id"]]
        )

def retrieve(query: str, filters: Optional[Dict[str, Any]] = None, top_k: int = 5) -> List[str]:
    assert _embedder is not None, "Embedder not initialized"
    q_emb = _embedder.encode(query).tolist()
    res = _collection.query(query_embeddings=[q_emb], n_results=top_k, where=filters or {})
    return res.get("documents", [[]])[0]
