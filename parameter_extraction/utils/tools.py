#!/usr/bin/env python3
from pathlib import Path
import json
from sentence_transformers import SentenceTransformer
import chromadb

# --- Default paths and settings ---
DEFAULT_COLLECTION = "parameters"
DEFAULT_PERSIST = "./memory/params_db"
DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"

# ==========================================================
# 📘 Embed Mode: Convert merged JSON parameters into embeddings
# ==========================================================
def embed_mode(pdf=None, collection=DEFAULT_COLLECTION, persist=DEFAULT_PERSIST, embed_model=DEFAULT_EMBED_MODEL):
    """
    Embeds all parameter instances from merged_parameters.json into a Chroma vector database.
    """

    merged_file = Path(__file__).resolve().parent.parent / "results" / "merged_parameters.json"

    if not merged_file.exists():
        raise FileNotFoundError(f"❌ Could not find merged JSON at {merged_file}")

    print(f"📂 Loading merged parameters from {merged_file}")
    with open(merged_file, "r", encoding="utf-8") as f:
        merged_data = json.load(f)

    parameters = merged_data.get("parameters", {})
    print(f"📊 Found {len(parameters)} parameters to embed")

    # --- Initialize embedding model ---
    print(f"🧠 Loading embedding model: {embed_model}")
    model = SentenceTransformer(embed_model)

    # --- Initialize Chroma client ---
    persist_path = Path(persist)
    persist_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_path))
    collection = client.get_or_create_collection(collection)

    # --- Build embedding documents ---
    docs, ids, metas = [], [], []
    for i, (param_name, data) in enumerate(parameters.items()):
        for j, inst in enumerate(data["instances"]):
            text = (
                f"{param_name} — value: {inst.get('value')}, "
                f"units: {inst.get('units')}, "
                f"context: {inst.get('context')}, "
                f"source: {inst.get('source')}, "
                f"section: {inst.get('section')}"
            )
            docs.append(text)
            ids.append(f"{param_name}_{j}")
            metas.append({
                "parameter": param_name,
                "source": inst.get("source"),
                "section": inst.get("section"),
                "context": inst.get("context"),
            })

    print(f"🧮 Encoding {len(docs)} parameter instances into embeddings...")
    embeddings = model.encode(docs, show_progress_bar=True, convert_to_numpy=True)

    print("💾 Adding embeddings to vector database...")
    collection.add(documents=docs, metadatas=metas, ids=ids, embeddings=embeddings)
    print("✅ Embedding complete. You can now use `--mode retrieve` to query the parameter database.")


# ==========================================================
# 🔍 Retrieve Mode: Query vector database for similar parameters
# ==========================================================
def retrieve_mode(query, topk=8, collection=DEFAULT_COLLECTION, persist=DEFAULT_PERSIST, embed_model=DEFAULT_EMBED_MODEL, **kwargs):
    """
    Retrieves parameter entries semantically similar to a query.
    """

    if not query:
        print("[!] Please provide a query string.")
        return

    persist_path = Path(persist)
    if not persist_path.exists():
        raise FileNotFoundError(f"❌ Vector DB not found at {persist_path}. Run embed mode first.")

    print(f"📁 Connecting to vector DB at {persist_path}")
    client = chromadb.PersistentClient(path=str(persist_path))
    collection = client.get_collection(collection)

    print(f"🔍 Retrieving top {topk} results for query: '{query}'")
    model = SentenceTransformer(embed_model)
    query_emb = model.encode([query], convert_to_numpy=True)

    results = collection.query(query_embeddings=query_emb, n_results=topk)
    if not results["documents"]:
        print("⚠️ No results found.")
        return

    print("\n=== Retrieval Results ===")
    for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0]), 1):
        print(f"{i:02d}. {meta.get('parameter')}")
        print(f"   value/context: {meta.get('context')}")
        print(f"   source: {meta.get('source')}  ({meta.get('section')})")
        print(f"   text: {doc[:120]}...")
        print()

    print("=========================\n")
    return results


# ==========================================================
# Stub for transcribe mode (not used in parameter embedding)
# ==========================================================
def transcribe_mode(pdf=None, dpi=300):
    print("⚠️ Transcription mode not applicable for parameter embeddings.")
