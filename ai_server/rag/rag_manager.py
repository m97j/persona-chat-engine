import os, json
from typing import List, Dict, Any, Optional
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import EmbeddingFunction

_client = PersistentClient(path="./rag")
_collection = _client.get_or_create_collection(name="game_docs")
_embedder: Optional[EmbeddingFunction] = None

def set_embedder(embedder: Any):
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
                    for i, doc in enumerate(data):
                        if "id" not in doc:
                            doc["id"] = f"{filename}_{i}"
                        docs.append(doc)
                else:
                    if "id" not in data:
                        data["id"] = filename
                    docs.append(data)
        elif filename.endswith(".txt"):
            with open(full, "r", encoding="utf-8") as f:
                content = f.read()
                docs.append({
                    "id": filename,
                    "content": content,
                    "metadata": {}
                })
    return docs

def add_docs(docs: List[Dict[str, Any]], batch_size: int = 32):
    assert _embedder is not None, "Embedder not initialized"
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i+batch_size]
        ids = []
        contents = []
        embeddings = []
        metadatas = []
        for doc in batch:
            assert "id" in doc and "content" in doc, "doc requires id and content"
            ids.append(doc["id"])
            contents.append(doc["content"])
            metadatas.append(doc.get("metadata", {}))
            emb = _embedder.encode(doc["content"]).tolist()
            embeddings.append(emb)
        _collection.add(
            documents=contents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

def retrieve(query: Optional[str] = None, filters: Optional[Dict[str, Any]] = None, top_k: int = 5) -> List[Dict[str, Any]]:
    assert _embedder is not None, "Embedder not initialized"
    
    if query:
        q_emb = _embedder.encode(query).tolist()
        res = _collection.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            where=filters or {}
        )
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        return [{"content": d, "metadata": m} for d, m in zip(docs, metas)]
    else:
        res = _collection.get(
            where=filters or {},
            limit=top_k
        )
        docs = res.get("documents", [])
        metas = res.get("metadatas", [])
        return [{"content": d, "metadata": m} for d, m in zip(docs, metas)]
