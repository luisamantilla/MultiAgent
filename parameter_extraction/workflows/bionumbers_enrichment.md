# BioNumbers Enrichment Workflow

This workflow adds a retrieval fallback that pulls quantitative values from the [BioNumbers](https://bionumbers.hms.harvard.edu/) knowledge base whenever the embedded `merged_parameters` vector store does not contain a requested parameter. It assumes the existing memory layer already stores embeddings for the local JSON corpus (e.g., the `parameter_extraction/memory/params_db` directory) and that the CLI has network access and an HTTP client available.

## Overview

1. **User query → vector DB lookup**
   - Accept a text query such as “What is the diffusion coefficient of IL-10?”.
   - Embed it using the same model that produced the existing parameter embeddings.
   - Retrieve the top‑`k` passages/parameters from the local vector DB. If the similarity score of the best match is above a configured threshold, return that response as usual.

2. **Confidence check**
   - If no passage clears the confidence cutoff, trigger the BioNumbers enrichment path.
   - Log the miss so you can monitor which concepts require more coverage.

3. **BioNumbers search**
   - Send the query text to the BioNumbers site. BioNumbers exposes a simple search endpoint that accepts URL parameters (`https://bionumbers.hms.harvard.edu/search.aspx?query=...`). Use an HTTP client (e.g., `requests`) to fetch the results HTML.
   - Parse the search result table to extract candidate entry IDs and short descriptions. Use lightweight HTML parsing (BeautifulSoup or lxml).
   - Optionally re-rank results with the same embedding model to keep only semantically relevant rows.

4. **Entry scrape**
   - For the top entry (or top `n`), fetch the detail page (`https://bionumbers.hms.harvard.edu/bionumber.aspx?s=q&v={ID}`).
   - Parse the structured elements:
     - `BNID`
     - Quantity description
     - Numerical value and units
     - Organism / system
     - Reference / PMID
   - Normalize numeric strings (convert to floats when possible) and clean units/context text.

5. **Augment local parameter store**
   - Build a parameter document using the same nested schema the merge script produces:
     ```json
     {
       "parameter": "Interleukin 10 diffusion coefficient",
       "instances": [
         {
           "value": "2e-11",
           "units": "m^2/s",
           "context": "BioNumbers BNID 111111, human tissue",
           "source": "bionumbers_BNID111111",
           "section": "external"
         }
       ]
     }
     ```
     (If the parameter already exists, append a new instance to its `instances` array.)
   - Append this record to:
     1. `parameter_extraction/results/merged_parameters.json` (either by reusing the merge script or by calling a helper that inserts parameters programmatically).
     2. The embedding database: compute an embedding for the textual representation (e.g., `"Interleukin 10 diffusion coefficient 2e-11 m^2/s ..."`), then upsert it into the vector store together with a pointer to the new JSON entry.

6. **Return result to the user**
   - Respond with the scraped value, clearly noting that it came from BioNumbers.
   - Cache the query → BNID mapping so repeated requests do not re-scrape the site.

7. **Background maintenance**
   - Periodically recompute embeddings for the merged database to keep local + BioNumbers entries aligned.
   - Store a changelog (`log/bionumbers_enrichment.log`) documenting inserted BNIDs and timestamps for reproducibility/citation.

## Suggested File Structure

```
parameter_extraction/
├── external_sources/
│   ├── bionumbers_client.py       # wraps HTTP fetch + parsing logic ✅ implemented
│   └── bionumbers_cache.json      # optional persistent cache of BNID records
├── workflows/
│   ├── bionumbers_enrichment.md   # (this document)
│   └── ...
├── utils/
│   └── vector_store.py            # helper to query/upsert embeddings ✅ implemented
└── agents/
    ├── bionumbers_agent.py        # orchestrates the flow described below ✅ implemented
    └── retrieval_agent.py
```

> **Reference implementations in this repo**
>
> - `merge_2_updated`: handles normalization/merging of JSON parameter pages. The enrichment agent can import and reuse its helper routines to append a freshly scraped entry into `merged_parameters.json` so the dataset stays consistent.
> - `resume_extractor.py`: demonstrates how the project structures OpenAI API calls and result handling. Reusing its HTTP/client scaffolding can accelerate the BioNumbers client implementation.

## Execution Flow (Pseudo-code)

```python
from agents.bionumbers_agent import BioNumbersRetrievalAgent

agent = BioNumbersRetrievalAgent()
result = agent.answer_parameter_query("What is the IL-10 diffusion coefficient?")
print("Local matches:")
for hit in result["local"]:
    print("  -", hit["parameter"], hit["instances"])
print("BioNumbers matches:")
for hit in result["bionumbers"]:
    print("  -", hit["parameter"], "(BNID", hit["bnid"], ")")
```

## What This Adds

- **Automatic coverage expansion**: Any miss in the internal database now triggers a BioNumbers lookup, reducing manual labor.
- **Consistent storage**: Newly found parameters are normalized to the same schema, saved locally, and indexed for future queries.
- **Auditable provenance**: Each addition keeps the BNID and citation, so downstream use retains proper sourcing.

Adapt the pseudo-code to whichever agent loop (CLI chatbot, API server, etc.) you are using. The important part is the clean separation between: retrieval attempt → external enrichment → local persistence.
