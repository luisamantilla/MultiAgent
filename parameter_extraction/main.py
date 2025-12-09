#!/usr/bin/env python3
import argparse
from pathlib import Path
from utils.tools import (
    transcribe_mode, embed_mode, retrieve_mode,
    DEFAULT_COLLECTION, DEFAULT_PERSIST, DEFAULT_EMBED_MODEL
)

def main():
    p = argparse.ArgumentParser(description="Transcribe / Embed / Retrieve")
    p.add_argument("--mode", choices=["transcribe", "embed", "retrieve"], required=True)

    # shared
    p.add_argument("--pdf", type=str)
    p.add_argument("--dpi", type=int, default=300)

    # embed/retrieve
    p.add_argument("--collection", type=str, default=DEFAULT_COLLECTION)
    p.add_argument("--persist", type=str, default=DEFAULT_PERSIST)
    p.add_argument("--embed_model", type=str, default=DEFAULT_EMBED_MODEL)

    # retrieve
    p.add_argument("--query", type=str)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--prefer_doc", type=str, default="")
    p.add_argument("--candidates", type=int, default=3)
    p.add_argument("--preview", action="store_true")
    p.add_argument("--no_expand_synonyms", action="store_true")

    args = p.parse_args()

    if args.mode == "transcribe":
        transcribe_mode(pdf=args.pdf, dpi=args.dpi)
    elif args.mode == "embed":
        embed_mode(pdf=args.pdf, collection=args.collection, persist=args.persist, embed_model=args.embed_model)
    else:
        retrieve_mode(
            query=args.query, topk=args.topk, collection=args.collection, persist=args.persist,
            embed_model=args.embed_model, prefer_doc=args.prefer_doc, candidates=args.candidates,
            preview=args.preview, no_expand_synonyms=args.no_expand_synonyms, out_file=None
        )

if __name__ == "__main__":
    main()