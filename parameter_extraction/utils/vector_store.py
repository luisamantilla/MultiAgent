"""
Vector store helpers backed by the persistent Chroma database located at
parameter_extraction/memory/params_db. This lets the CLI and agents reuse the
embeddings produced via utils/tools.py (SentenceTransformer + Chroma).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import uuid

import chromadb
from sentence_transformers import SentenceTransformer


MEMORY_DIR = Path("parameter_extraction/memory/params_db")
COLLECTION_NAME = "parameters"
EMBED_MODEL = "all-MiniLM-L6-v2"
MERGED_FILE = Path("parameter_extraction/results/merged_parameters.json")

_MODEL: Optional[SentenceTransformer] = None
_MERGED_CACHE: Optional[Dict[str, Any]] = None


def get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(EMBED_MODEL)
    return _MODEL


def get_collection():
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(MEMORY_DIR))
    return client.get_or_create_collection(COLLECTION_NAME)


def load_merged() -> Dict[str, Any]:
    global _MERGED_CACHE
    if _MERGED_CACHE is None:
        if MERGED_FILE.exists():
            _MERGED_CACHE = json.loads(MERGED_FILE.read_text(encoding="utf-8"))
        else:
            _MERGED_CACHE = {"parameters": {}}
    return _MERGED_CACHE


def query_vector_db(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """Return the top-k local matches from Chroma, joined with merged JSON."""
    query = (query or "").strip()
    if not query:
        return []
    collection = get_collection()
    if collection.count() == 0:
        return []
    model = get_model()
    query_emb = model.encode([query], convert_to_numpy=True)
    results = collection.query(query_embeddings=query_emb, n_results=k)
    if not results or not results.get("metadatas"):
        return []

    merged = load_merged().get("parameters", {})
    ids = results.get("ids", [[]])[0]
    distances = results.get("distances", [[]])[0]
    metas = results["metadatas"][0]

    output: List[Dict[str, Any]] = []
    for idx, meta in enumerate(metas):
        param_name = meta.get("parameter")
        if not param_name:
            continue
        payload = merged.get(param_name, {"instances": []})
        score = distance_to_similarity(distances[idx] if idx < len(distances) else None)
        output.append(
            {
                "id": ids[idx] if idx < len(ids) else None,
                "score": score,
                "parameter": param_name,
                "instances": payload.get("instances", []),
            }
        )
    return output


def upsert_record(record: Dict[str, Any]) -> None:
    """Insert new instances into the Chroma collection."""
    parameter = record.get("parameter")
    instances = record.get("instances", [])
    if not parameter or not instances:
        return
    collection = get_collection()
    model = get_model()
    documents: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    ids: List[str] = []
    for inst in instances:
        documents.append(build_instance_text(parameter, inst))
        metadatas.append(
            {
                "parameter": parameter,
                "source": inst.get("source"),
                "section": inst.get("section"),
                "context": inst.get("context"),
            }
        )
        ids.append(f"{parameter}_{uuid.uuid4().hex}")
    embeddings = model.encode(documents, convert_to_numpy=True)
    collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)


def build_instance_text(parameter: str, inst: Dict[str, Any]) -> str:
    return (
        f"{parameter} — value: {inst.get('value')}, "
        f"units: {inst.get('units')}, "
        f"context: {inst.get('context')}, "
        f"source: {inst.get('source')}, "
        f"section: {inst.get('section')}"
    )


def distance_to_similarity(distance: Optional[float]) -> Optional[float]:
    if distance is None:
        return None
    return 1.0 / (1.0 + float(distance))
