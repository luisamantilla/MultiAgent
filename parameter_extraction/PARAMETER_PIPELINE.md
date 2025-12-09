# Parameter Extraction & BioNumbers Enrichment Pipeline

This document summarizes the full workflow for building and querying the parameter knowledge base, including the BioNumbers fallback. All referenced scripts live inside `parameter_extraction/` so you can run them directly from the repo root.

## 1. Source Extraction (`resume_extractor.py`)

1. Drop PDFs / text files into `parameter_extraction/Tumor_tcell_papers_text/`.
2. Run `python parameter_extraction/resume_extractor.py`.
3. The script uses `utils/vision_based_processor.py` to render each PDF page or text snippet, call GPT-4o, and save per-page JSON output under `parameter_extraction/results/<stem>_results/`.
4. Processed source files are moved to `_processed` to avoid duplicate work. The `merged_parameters.json` file tracks what has already been ingested.

## 2. Merge & Deduplicate (`merge_2_updated`)

1. Execute `python parameter_extraction/merge_2_updated`.
2. The script walks every result JSON, consolidates parameter dictionaries, deduplicates identical instances, and writes the canonical database to `parameter_extraction/results/merged_parameters.json`.
3. Each parameter entry follows the nested schema:
   ```json
   "Parameter Name": {
     "instances": [
       {
         "value": "...",
         "units": "...",
         "context": "...",
         "source": "20251006_094742_page5.json",
         "section": "tables"
       }
     ]
   }
   ```

## 3. Embedding the Database (`utils/vector_store.py`)

1. Embeddings are stored in a persistent Chroma collection at `parameter_extraction/memory/params_db`, populated via `utils/tools.py --mode embed` (SentenceTransformer `all-MiniLM-L6-v2`).
2. `query_vector_db` connects to that Chroma store, encodes the query with the same model, and returns the top‑k parameter names plus their full instances (looked up from `merged_parameters.json`).
3. `upsert_record` adds new BioNumbers instances into both Chroma and the merged JSON so future queries immediately see the enriched data.

## 4. Command-line Query (`query_parameter.py`)

Run:
```
python parameter_extraction/query_parameter.py "IL-10 diffusion coefficient"
```

The script:
1. Instantiates `BioNumbersRetrievalAgent` with configurable `--local-k` and `--bionumbers-k`.
2. Always retrieves the top-k local matches from the embedding store and prints their scores plus instances.
3. Independently queries BioNumbers for the same text, fetches the top-k BNIDs, normalizes them, persists them (unless `--no-persist`), and prints the results.
4. Output includes two blocks (“Local DB matches” and “BioNumbers matches”) so you can compare both sources side by side in one command.

## 5. BioNumbers Enrichment

- BioNumbers search/scrape logic lives in `external_sources/bionumbers_client.py`.
- Each BNID result is converted to the nested schema via `BioNumberEntry.to_parameter_doc()`.
- `BioNumbersRetrievalAgent` persists every fetched instance into `merged_parameters.json` (avoiding duplicates) and calls `upsert_record` so the embedding store expands automatically. Disable persistence with `--no-persist` if you need a read-only lookup.
- CLI output annotates BioNumbers hits with the BNID so you can cite the source.

## 6. Files Involved

- `resume_extractor.py` – orchestrates GPT extraction for PDFs/text.
- `merge_2_updated` – merges extracted JSON into the canonical database.
- `results/merged_parameters.json` – single source of truth.
- `utils/vector_store.py` – embeddings + similarity search.
- `external_sources/bionumbers_client.py` – BioNumbers HTTP & parsing.
- `agents/bionumbers_agent.py` – local-query + fallback orchestration.
- `query_parameter.py` – CLI entry point.
- `workflows/bionumbers_enrichment.md` – deeper design notes.

With these components you can ingest new papers, keep the embedded store synced, and transparently enrich gaps from BioNumbers—all from scripts housed in the `parameter_extraction` folder.

---

## Troubleshooting

### ChromaDB Corruption Error

If you encounter a Rust panic error like:
```
thread '<unnamed>' panicked at rust/sqlite/src/db.rs:157:42:
range start index 10 out of range for slice of length 9
```

This indicates ChromaDB corruption. To fix it:

1. **Navigate to project root:**
   ```bash
   cd /path/to/your/project
   ```

2. **Remove the corrupted database:**
   ```bash
   rm -rf parameter_extraction/memory/params_db
   ```

3. **Rebuild the vector store from your merged parameters:**
   ```bash
   python parameter_extraction/main.py --mode embed --persist parameter_extraction/memory/params_db
   ```

4. **Test the query:**
   ```bash
   python parameter_extraction/query_parameter.py "your query here"
   ```

**Note:** If the error persists after rebuilding, try:
- Clearing Python cache: `find parameter_extraction -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null`
- Restarting your terminal to clear any in-memory ChromaDB cache
- Upgrading ChromaDB: `pip install --upgrade chromadb`

### BioNumbers Returns No Results

If BioNumbers queries return empty results despite successful HTTP responses:

1. The BioNumbers website structure may have changed. Check `external_sources/bionumbers_client.py` for:
   - Search URL parameters (currently uses `trm`)
   - HTML parsing selectors (currently looks for `<a href="bionumber.aspx?id=...">`)
   - Entry page structure (currently uses `<h1>` for name, `<th>`/`<td>` for fields)

2. Test the client directly:
   ```bash
   python -c "
   import sys
   sys.path.insert(0, 'parameter_extraction')
   from external_sources.bionumbers_client import BioNumbersClient
   client = BioNumbersClient()
   print(client.search('glucose', max_results=3))
   "
   ```

### Slow Query Performance

If queries are taking too long:

1. **Model loading (~4s)**: The SentenceTransformer model loads on first use. This is normal and only happens once per Python process.

2. **Vector query (~2s)**: Increase `--local-k` or reduce collection size if searches are consistently slow.

3. **BioNumbers (~2s per entry)**: Reduce `--bionumbers-k` to fetch fewer external results.
