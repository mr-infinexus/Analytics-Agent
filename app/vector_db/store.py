import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
import httpx

from app.config import settings


class GeminiEmbeddingFunction(EmbeddingFunction[Documents]):
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY", "")
        self.model = model or settings.GEMINI_EMBEDDING_MODEL

    @staticmethod
    def name() -> str:
        return "gemini_embedding_2_preview"

    def get_config(self) -> Dict[str, Any]:
        return {"model": self.model}

    @classmethod
    def build_from_config(cls, config: Dict[str, Any]) -> "GeminiEmbeddingFunction":
        return cls(model=config.get("model"))

    def __call__(self, input: Documents) -> Embeddings:
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY required for vector embeddings.")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:batchEmbedContents?key={self.api_key}"
        payload = [
            {"model": f"models/{self.model}", "content": {"parts": [{"text": text}]}}
            for text in input
        ]
        with httpx.Client(timeout=30.0) as client:
            res = client.post(url, json={"requests": payload})
            res.raise_for_status()
            data = res.json()
            return [item["values"] for item in data.get("embeddings", [])]


class VectorStore:
    def __init__(self, persist_dir: Optional[str] = None):
        self.persist_path = settings.resolve_path(persist_dir or settings.CHROMA_PERSIST_DIR)
        self.persist_path.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=str(self.persist_path))
        self.embedding_fn = GeminiEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name="analytics_few_shot_examples",
            embedding_function=self.embedding_fn,
        )

    def seed_from_file(self, file_path: Path):
        """Seed few-shot examples from JSON file into ChromaDB if not already seeded."""
        if not file_path.exists() or self.collection.count() > 0:
            return

        with open(file_path, "r", encoding="utf-8") as f:
            examples = json.load(f)

        ids, docs, metas = [], [], []
        for idx, item in enumerate(examples):
            q = item.get("query", "").strip()
            logic = item.get("expected_logic", "").strip()
            if q:
                ids.append(f"ex_{idx}")
                docs.append(q)
                metas.append({"query": q, "expected_logic": logic})

        if ids:
            self.collection.upsert(ids=ids, documents=docs, metadatas=metas)

    def get_similar_examples(self, query: str, n: int = 3) -> List[Dict[str, Any]]:
        """Retrieve top-n similar example queries for few-shot prompt context."""
        count = self.collection.count()
        if count == 0:
            return []

        results = self.collection.query(
            query_texts=[query],
            n_results=min(n, count),
        )
        if results and results.get("metadatas") and results["metadatas"][0]:
            return [dict(m) for m in results["metadatas"][0]]
        return []


_vector_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
