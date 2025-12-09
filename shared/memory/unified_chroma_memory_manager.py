import os
import json
import chromadb
from chromadb.config import Settings
from openai import OpenAI
from typing import Optional, List, Dict, Any

class UnifiedChromaMemoryManager:
    """
    Combines structured agent memory (iteration, agent, summary, output_file) with semantic search using OpenAI embeddings and ChromaDB.
    Enforces a 512-token (approx 2000 char) summary limit for both storage and embedding.
    """
    def __init__(self, db_dir: str, collection_name: str = "agent_memory", embedding_model: str = "text-embedding-3-large"):
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.embedding_model = embedding_model
        self.chroma_client = chromadb.PersistentClient(path=db_dir)
        self.collection = self.chroma_client.get_or_create_collection(collection_name)

    def get_embedding(self, text: str) -> List[float]:
        # Enforce 512-token (approx 2000 char) limit for embedding
        text = text[:2000]
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=[text]
        )
        return response.data[0].embedding

    def write_memory(self, iteration: int, agent: str, summary: str, output_file: Optional[str] = None) -> str:
        # Enforce 512-token (approx 2000 char) limit for summary and embedding
        summary = summary[:2000]
        memory = {
            "iteration": iteration,
            "agent": agent,
            "summary": summary,
            "output_file": output_file or ""
        }
        memory_id = f"{agent}-{iteration}"
        embedding = self.get_embedding(summary)
        self.collection.upsert(
            embeddings=[embedding],
            documents=[json.dumps(memory)],
            metadatas=[{"iteration": iteration, "agent": agent}],
            ids=[memory_id]
        )
        return memory_id

    def get_latest_memory(self, agent: Optional[str] = None) -> Optional[Dict[str, Any]]:
        results = self.collection.get(include=["documents", "metadatas"])
        docs = results.get("documents", [])
        metas = results.get("metadatas", [])
        filtered = [json.loads(doc) for doc, meta in zip(docs, metas) if (agent is None or meta.get("agent") == agent)]
        if not filtered:
            return None
        return max(filtered, key=lambda m: m["iteration"])

    def get_memories(self, agent: Optional[str] = None, iteration: Optional[int] = None, keyword: Optional[str] = None) -> List[Dict[str, Any]]:
        results = self.collection.get(include=["documents", "metadatas"])
        docs = results.get("documents", [])
        metas = results.get("metadatas", [])
        filtered = []
        for doc, meta in zip(docs, metas):
            m = json.loads(doc)
            if agent and m["agent"] != agent:
                continue
            if iteration is not None and m["iteration"] != iteration:
                continue
            if keyword and keyword.lower() not in m["summary"].lower():
                continue
            filtered.append(m)
        return filtered

    def semantic_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        embedding = self.get_embedding(query)
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k
        )
        docs = results['documents'][0] if results['documents'] else []
        return [json.loads(doc) for doc in docs]

    def print_all_memories(self):
        memories = self.get_memories()
        for m in sorted(memories, key=lambda x: (x["iteration"], x["agent"])):
            print(json.dumps(m, indent=2))
