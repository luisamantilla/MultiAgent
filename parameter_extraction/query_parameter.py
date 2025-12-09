#!/usr/bin/env python3
"""
CLI helper to query the local parameter database with automatic BioNumbers
fallback. Example usage:

    python parameter_extraction/query_parameter.py "IL-10 diffusion coefficient"
"""
import argparse
import sys
import time

from agents.bionumbers_agent import BioNumbersRetrievalAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query parameter DB + BioNumbers")
    parser.add_argument("query", nargs="*", help="Parameter description to search for")
    parser.add_argument("--local-k", type=int, default=5, help="Top-k local matches to display")
    parser.add_argument("--bionumbers-k", type=int, default=3, help="Top-k BioNumbers hits to fetch")
    parser.add_argument(
        "--no-persist",
        action="store_true",
        help="Do not persist BioNumbers hits back into merged_parameters.json",
    )
    return parser.parse_args()


def format_instance(inst: dict) -> str:
    value = inst.get("value")
    units = inst.get("units")
    context = inst.get("context")
    source = inst.get("source")
    parts = []
    if value is not None:
        parts.append(str(value))
    if units:
        parts.append(str(units))
    if context:
        parts.append(f"[{context}]")
    if source:
        parts.append(f"(source: {source})")
    return " ".join(parts)


def main() -> None:
    args = parse_args()
    query = " ".join(args.query).strip()
    if not query:
        print("Enter a parameter query (e.g., 'IL-10 diffusion coefficient').")
        return

    start_time = time.time()

    print("⏳ Initializing agent (loading embedding model)...")
    sys.stdout.flush()
    init_start = time.time()
    agent = BioNumbersRetrievalAgent(
        local_k=args.local_k,
        bionumbers_k=args.bionumbers_k,
        persist_bionumbers=not args.no_persist,
    )
    print(f"   ✓ Done ({time.time() - init_start:.1f}s)")
    sys.stdout.flush()

    print("🔍 Searching local database...")
    sys.stdout.flush()
    local_start = time.time()
    local_hits = agent.get_local_hits(query, k=args.local_k)
    print(f"   ✓ Found {len(local_hits)} matches ({time.time() - local_start:.1f}s)")
    sys.stdout.flush()

    print("🌐 Querying BioNumbers...")
    sys.stdout.flush()
    bn_start = time.time()
    bionumbers_hits = agent.get_bionumbers_hits(
        query, k=args.bionumbers_k, persist=not args.no_persist
    )
    print(f"   ✓ Found {len(bionumbers_hits)} matches ({time.time() - bn_start:.1f}s)")
    sys.stdout.flush()

    results = {"local": local_hits, "bionumbers": bionumbers_hits}
    print(f"\n✅ Total time: {time.time() - start_time:.1f}s\n")
    sys.stdout.flush()

    print(f"🔎 Query: {query}")
    print("\nLocal DB matches:")
    if not local_hits:
        print("  (none)")
    else:
        for idx, hit in enumerate(local_hits, start=1):
            score = hit.get("score")
            score_text = f"{score:.3f}" if isinstance(score, (int, float)) else "n/a"
            print(f"  {idx}. {hit['parameter']}  [score={score_text}]")
            for inst in hit.get("instances", []):
                print("     -", format_instance(inst))

    print("\nBioNumbers matches:")
    if not bionumbers_hits:
        print("  (none)")
    else:
        for idx, hit in enumerate(bionumbers_hits, start=1):
            print(f"  {idx}. {hit['parameter']}  (BNID {hit['bnid']})")
            for inst in hit.get("instances", []):
                print("     -", format_instance(inst))


if __name__ == "__main__":
    main()
