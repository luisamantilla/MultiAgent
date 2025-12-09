from collections import OrderedDict, defaultdict
import json
from math import ceil
import os
import re
from dotenv import load_dotenv
import pandas as pd
from openai import OpenAI

from qdrant_client import QdrantClient
from qdrant_client.models import NamedVector, SearchParams

from demo.db.vector.vector_db import embed


load_dotenv()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION_URL = os.getenv("QDRANT_COLLECTION_URL")
OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")

QDRANT_CLIENT = QdrantClient(
    url=QDRANT_COLLECTION_URL, 
    api_key=QDRANT_API_KEY,
)
OPEN_AI_CLIENT = OpenAI(api_key=OPEN_AI_API_KEY)

PARAGRAPH_ARTICLE_LIMIT = 4
QUERY_VARIANTS = [
    "PD1 positive T cell death rate",
    "Annexin V apoptosis percentage",
    "T cell proliferation rate",
    "cytokine concentration pg/ml",
    "tumor size reduction mm",
    "IC50 value cancer cells",
]



def _fetch_pool(query: str, num_points: int) -> list:
    variants = QUERY_VARIANTS
    pool_mult = 4  # widen recall ~4x your desired N
    limit_per = max(100, ceil((num_points * pool_mult) / len(variants)))
    ef = 512

    combined = OrderedDict()
    for v in variants:
        vec = embed(v)
        hits = QDRANT_CLIENT.search(
            collection_name="pubmed_articles",
            query_vector=NamedVector(name="embedding", vector=vec),
            limit=limit_per,
            with_payload=True,
            search_params=SearchParams(hnsw_ef=ef)
        )
        for h in hits:
            # dedupe by point id; fall back to (doi, para_idx) if needed
            key = getattr(h, "id", None)
            if key is None:
                pl = h.payload or {}
                key = (pl.get("doi"), pl.get("para_idx"), hash(pl.get("p", "")))
            if key not in combined:
                combined[key] = h
    return list(combined.values())

def generate_data(query: str, num_points: int = 100) -> pd.DataFrame:
    regex = determine_measurement_type_regex(query)

    # 1) fetch a widened, deduped pool across variants
    pool_hits = _fetch_pool(query, num_points)

    # 2) score + enforce per-article paragraph cap
    results_by_doi = defaultdict(list)
    for hit in pool_hits:
        doi = hit.payload.get("doi")
        if not doi: 
            continue
        if len(results_by_doi[doi]) >= PARAGRAPH_ARTICLE_LIMIT:
            continue
        
        paragraph = hit.payload.get("p", "")
        
        point = {
            "doi": doi,
            "section": hit.payload.get("section"),
            "numeric_score": hit.payload.get("numeric_score", 0),
            "measurement_type_score": 1 if run_regex_measurement_types(determine_measurement_type_regex(query), paragraph) else 0,
            "cosine_score": hit.score,
            "paragraph": paragraph
        }
        results_by_doi[doi].append(point)
        
    # Per DOI data
    rows = []
    for doi, points in results_by_doi.items():
        max_cos = max(float(p["cosine_score"]) for p in points) if points else 0.0
        max_num = max(float(p.get("numeric_score") or 0.0) for p in points) if points else 0.0
        max_measure = any(int(p.get("measurement_type_score", 0)) == 1 for p in points)
        
        
        rows.append({
            ""
        })

    # 3) flatten to rows → DataFrame
    rows = [pt for group in results_by_doi.values() for pt in group]
    return pd.DataFrame(rows)

    
def run_regex_measurement_types(regex: list[str], paragraph: str) -> bool:
    for r in regex:
        if(bool(re.search(r, paragraph))):
            return True
    return False

def determine_measurement_type_regex(query: str):
    """
    Returns an OpenAI assistant that **only** converts parameter
    descriptions into PubMed / PMC Boolean search queries.
    """
    agent = OPEN_AI_CLIENT.beta.assistants.create(
        name="Experiment Regex Generator",
        model="gpt-4o",
        tools=[],
        instructions=(
            "You output ONLY a JSON array of regex strings. No prose, no keys, no labels.\n"
            "INPUT: a parameter phrase (e.g., 'T-cell half-life', 'CD8+ death rate', 'IFNγ secretion rate', 'KD for PD-1:PD-L1').\n"
            "TASK: Infer which experimental/simulation methods are typically used to measure THAT parameter. "
            "Then emit 5–15 regex patterns—one per method—covering common names and synonyms for each method.\n"
            "If no common experimental/simulation methods exist for measuring THAT parameter, output only the word NONE."
            "\n"
            "RULES:\n"
            "• Regex must be Python 're'; case-insensitive via (?i); use word boundaries where sensible; combine synonyms with (?:...|...). "
            "• Keep patterns focused on method/assay names (no numeric extraction). Keep them short and specific.\n"
            "• Output ONLY a flat JSON array of strings. Nothing else.\n"
            "• Example of a single item shape: \"(?i)\\\\b(?:ELISpot|enzyme-?linked\\\\s+immuno\\\\s*spot)\\\\b\".\n"
        )
    )

    # Create a thread and run
    thread = OPEN_AI_CLIENT.beta.threads.create()
    run = OPEN_AI_CLIENT.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=agent.id,
        instructions=None,
        additional_messages=[{"role": "user", "content": query}]
    )

    # Wait for completion
    while True:
        run_status = OPEN_AI_CLIENT.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
        if run_status.status == "completed":
            break
        elif run_status.status in ("failed", "cancelled"):
            raise RuntimeError(f"Run failed with status: {run_status.status}")

    # Get result message
    messages = OPEN_AI_CLIENT.beta.threads.messages.list(thread_id=thread.id)
    raw =  messages.data[0].content[0].text.value
    
    return json.loads(raw)
    
def determine_measurement_type_regex(query: str):
    """
    Returns an OpenAI assistant that **only** converts parameter
    descriptions into PubMed / PMC Boolean search queries.
    """
    agent = OPEN_AI_CLIENT.beta.assistants.create(
        name="Experiment Regex Generator",
        model="gpt-4o",
        tools=[],
        instructions=(
            "You output ONLY a JSON array of regex strings. No prose, no keys, no labels.\n"
            "INPUT: a parameter phrase (e.g., 'T-cell half-life', 'CD8+ death rate', 'IFNγ secretion rate', 'KD for PD-1:PD-L1').\n"
            "TASK: Infer which experimental/simulation methods are typically used to measure THAT parameter. "
            "Then emit 5–15 regex patterns—one per method—covering common names and synonyms for each method.\n"
            "If no common experimental/simulation methods exist for measuring THAT parameter, output only the word NONE."
            "\n"
            "RULES:\n"
            "• Regex must be Python 're'; case-insensitive via (?i); use word boundaries where sensible; combine synonyms with (?:...|...). "
            "• Keep patterns focused on method/assay names (no numeric extraction). Keep them short and specific.\n"
            "• Output ONLY a flat JSON array of strings. Nothing else.\n"
            "• Example of a single item shape: \"(?i)\\\\b(?:ELISpot|enzyme-?linked\\\\s+immuno\\\\s*spot)\\\\b\".\n"
        )
    )

    # Create a thread and run
    thread = OPEN_AI_CLIENT.beta.threads.create()
    run = OPEN_AI_CLIENT.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=agent.id,
        instructions=None,
        additional_messages=[{"role": "user", "content": query}]
    )

    # Wait for completion
    while True:
        run_status = OPEN_AI_CLIENT.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
        if run_status.status == "completed":
            break
        elif run_status.status in ("failed", "cancelled"):
            raise RuntimeError(f"Run failed with status: {run_status.status}")

    # Get result message
    messages = OPEN_AI_CLIENT.beta.threads.messages.list(thread_id=thread.id)
    raw =  messages.data[0].content[0].text.value
    
    return json.loads(raw)

def determine_parameter_extraction_score(paragraph: str, query: str) -> dict:
    agent = OPEN_AI_CLIENT.beta.assistants.create(
        name="Paragraph Extractability Labeler",
        model="gpt-4o",
        tools=[],
        instructions=(
            "You are ExtractabilityLabeler. For the given biomedical paragraph, decide if the given model parameter can be extracted.\n\n"
            "Labels:\n"
            "0 = Not extractable: no usable numeric parameter; numbers are irrelevant (n, p-values) or context too missing to map.\n"
            "1 = Potentially extractable: quantitative claim exists but a key piece is missing/ambiguous (unit/entity/context/assay/timepoint) or requires derivation/from figure only.\n"
            "2 = Definitely extractable: explicit parameter value or tight range with unit and enough context to map to the model (entity + condition).\n\n"
            "Output format:\n"
            "Return exactly one single-line JSON object with:\n"
            '{"label": <0|1|2>, "justification": "<≤25 words, cite key evidence or why missing>"}\n'
            "No code fences, no extra keys, no newlines, no trailing text. "
            "If uncertain, choose the lower label. Do not invent numbers or units.\n\n"
            "Examples:\n"
            "Input: Annexin V at 24 h showed CD8+ apoptosis of 0.12 h⁻¹ in B16 tumors (n=5).\n"
            'Output: {"label": 2, "justification": "Explicit rate with unit, entity (CD8+), assay (Annexin V), tumor model (B16), timepoint (24 h)."}\n'
            "Input: Annexin V indicated increased CD8+ apoptosis after anti-PD1.\n"
            'Output: {"label": 1, "justification": "Quantitative claim without numeric value or unit; assay named but value missing."}\n'
            "Input: n=12 mice; p=0.03; see Fig. 2.\n"
            'Output: {"label": 0, "justification": "Only sample size and p-value; no parameter value or mappable context."}'
        )
    )
    
    # Create a thread and run
    thread = OPEN_AI_CLIENT.beta.threads.create()
    run = OPEN_AI_CLIENT.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=agent.id,
        instructions=None,
        additional_messages=[{"role": "user", "content": query}, {"role": "user", "content": paragraph}]
    )

    # Wait for completion
    while True:
        run_status = OPEN_AI_CLIENT.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
        if run_status.status == "completed":
            break
        elif run_status.status in ("failed", "cancelled"):
            raise RuntimeError(f"Run failed with status: {run_status.status}")

    # Get result message
    messages = OPEN_AI_CLIENT.beta.threads.messages.list(thread_id=thread.id)
    raw =  messages.data[0].content[0].text.value
    
    return json.loads(raw)


if __name__ == "__main__":
    generate_data("PD1+ death rate", 300)