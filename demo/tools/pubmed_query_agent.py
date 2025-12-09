import os
from openai import OpenAI
from qdrant_client import QdrantClient


QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION_URL = os.getenv("QDRANT_COLLECTION_URL")
OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")

QDRANT_CLIENT = QdrantClient(
    url=QDRANT_COLLECTION_URL, 
    api_key=QDRANT_API_KEY,
)
OPEN_AI_CLIENT = OpenAI(api_key=OPEN_AI_API_KEY)

def pmc_query_agent():
    """
    Returns an OpenAI assistant that **only** converts parameter
    descriptions into PubMed / PMC Boolean search queries.
    """
    return OPEN_AI_CLIENT.beta.assistants.create(
        name="PMC Query-Writer",
        instructions = (
            "You are a biomedical search assistant.\n"
            "INPUT: a parameter (e.g., IFNγ or death rate) and a cell type.\n"
            "TASK: Output **one general** and **two specific** PubMed / PMC Boolean queries related to that parameter.\n"
            "\n"
            "RULES:\n"
            " • The **general query** should remove any specific markers or subtypes (e.g., PD-1+, CD8+) and use broad terms (e.g., 'T cell').\n"
            " • The **specific queries** should keep the detailed subtype (e.g., 'PD1+', 'CD8+') and expand common synonyms.\n"
            " • For all queries, include numeric-related terms like rate, concentration, half-life, kinetics, etc.\n"
            " • Use AND/OR logic and quotes as needed.\n"
            " • Output the three queries on one line, separated by commas—no extra text.\n"
            "\n"
            "EXAMPLE:\n"
            "INPUT  : \"IFN-γ secretion, CD8+ T cell\"\n"
            "OUTPUT : "
            "(\"IFNγ\" OR \"IFN-gamma\") AND (secretion OR release OR production) AND (rate OR kinetics) AND (\"T cell\"), "
            "(\"IFNγ\" OR \"IFN-gamma\") AND (secretion OR release) AND (rate OR half-life) AND (\"CD8+ T cell\"), "
            "(\"IFNγ\" OR \"IFN-gamma\") AND (secretion) AND (turnover OR abundance) AND (\"CD8+ T cell\")"
        ),
        tools=[],
        model="gpt-4o"
    )

def get_pmc_queries(parameter: str, cell_type: str):
    agent = pmc_query_agent()
    user_input = f"{parameter}, {cell_type}"

    # Create a thread and run
    thread = OPEN_AI_CLIENT.beta.threads.create()
    run = OPEN_AI_CLIENT.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=agent.id,
        instructions=None,
        additional_messages=[{"role": "user", "content": user_input}]
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
    return messages.data[0].content[0].text.value.strip()
