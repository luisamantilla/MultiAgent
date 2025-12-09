# tools/tools.py
import os
import json
import uuid
import requests
import urllib.parse
from openai import OpenAI
import tiktoken
import tempfile
import importlib.util
from vivarium.core.engine import Engine
from vivarium.core.composer import Composite
import xml.etree.ElementTree as ET
from typing import Optional, List

from demo.db.vector.vector_db import embed

from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue

PUBMED_TOOL_NAME = "pubmed_search"

PUBMED_TOOL_DESCRIPTION = {
    "type": "function",
    "function": {
        "name": PUBMED_TOOL_NAME,
        "description": "Retrieve articles from PubMed Central based on a biomedical query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query for PubMed articles."},
                "num_articles": {"type": "integer", "description": "Number of articles to retrieve."},
                "abstract_only": {"type": "boolean", "description": "Whether to return only abstracts."},
            },
            "required": ["query", "num_articles"]
        }
    }
}

def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def get_pubmed_central_article(pmcid: str, abstract_only: bool = False, max_tokens: int = 1000):
    # Construct the OAI-PMH URL
    url = f"https://www.ncbi.nlm.nih.gov/pmc/oai/oai.cgi?verb=GetRecord&identifier=oai:pubmedcentral.nih.gov:{pmcid}&metadataPrefix=pmc"

    try:
        response = requests.get(url)
        response.raise_for_status()
        root = ET.fromstring(response.content)
    except (requests.RequestException, ET.ParseError):
        return None, None

    # Navigate to the full-text body in JATS/NXML format
    ns = {
        'oai': 'http://www.openarchives.org/OAI/2.0/',
        'pmc': 'http://dtd.nlm.nih.gov/2.0/xsd/archivearticle'
    }

    # Extract the <article-title>
    title = root.find('.//pmc:article-title', ns)
    title_text = title.text if title is not None else None

    # Extract paragraphs
    passages: List[str] = []
    token_total = 0

    if abstract_only:
        paras = root.findall('.//pmc:abstract//pmc:p', ns)
        
        print(paras)
    else:
        # Restrict to conclusion/discussion/results sections
        paras = root.findall('.//pmc:sec', ns)
        filtered = []
        for sec in paras:
            title_elem = sec.find('pmc:title', ns)
            if title_elem is not None and title_elem.text:
                sec_title = title_elem.text.strip().lower()
                if any(key in sec_title for key in ["concl", "discuss", "result"]):
                    filtered.extend(sec.findall('pmc:p', ns))
        paras = filtered

    for p in paras:
        if p.text:
            text = p.text.strip()
            token_count = count_tokens(text)
            if token_total + token_count > max_tokens:
                break
            passages.append(text)
            token_total += token_count

    return title_text, passages


def run_pubmed_search(query: str, num_articles: int = 1, abstract_only: bool = True) -> str:
    print(f"[PubMed Search] Searching for: {query}")

    search_url = (
        f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?"
        f"db=pmc&term={urllib.parse.quote_plus(query)}&retmax={2 * num_articles}&retmode=json"
    )

    try:
        response = requests.get(search_url)
        response.raise_for_status()
        pmcids = response.json()["esearchresult"]["idlist"]
    except Exception as e:
        return f"PubMed query failed: {e}"

    articles = []

    for pmcid in pmcids:
        if len(articles) >= num_articles:
            break

        title, content = get_pubmed_central_article(pmcid, abstract_only=abstract_only)
        if not title or not content:
            continue

        print(f"[PubMed Article] PMCID: {pmcid}\nTitle: {title}\n---")
        article_text = f"PMCID: {pmcid}\nTitle: {title}\n\n" + "\n".join(content)
        articles.append(article_text)

    if not articles:
        return f"No articles found for: '{query}'."

    return "\n\n---\n\n".join(articles)


def run_tools_from_llm(run) -> list[dict[str, str]]:
    """
    Extract tool calls from a tool-using LLM run, and return their results.

    :param run: A chat completion run with pending tool calls
    :return: List of outputs in the form: [{"tool_call_id": ..., "output": ...}]
    """
    tool_outputs = []

    for tool in run.required_action.submit_tool_outputs.tool_calls:
        if tool.function.name == PUBMED_TOOL_NAME:
            args = json.loads(tool.function.arguments)
            output = run_pubmed_search(**args)
            tool_outputs.append({
                "tool_call_id": tool.id,
                "output": output
            })
        else:
            raise ValueError(f"Unknown tool: {tool.function.name}")

    return tool_outputs

def call_biologist_with_pubmed_via_chat(question: str) -> str:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # Step 1: Let assistant decide on a tool
    initial_response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a Biologist Agent with access to PubMed."},
            {"role": "user", "content": question}
        ],
        tools=[PUBMED_TOOL_DESCRIPTION],
        tool_choice="auto"
    )

    choice = initial_response.choices[0]
    if not hasattr(choice.message, "tool_calls"):
        return choice.message.content

    # Tool was called
    tool_call = choice.message.tool_calls[0]
    args = json.loads(tool_call.function.arguments)
    tool_output = run_pubmed_search(**args)

    # Step 2: Send the tool output + include assistant's tool_calls message
    follow_up = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a Biologist Agent with access to PubMed."},
            {"role": "user", "content": (
                "Only output executable Python logic suitable for use in the `next_update()` method "
                "of a Vivarium Process. Do not describe. Do not use Markdown. Only return code."
            )},
            {"role": "user", "content": question},
            {
                "role": "assistant",
                "tool_calls": [tool_call.model_dump()]
            },
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_call.function.name,
                "content": tool_output
            }
        ]
    )

    return follow_up.choices[0].message.content

# ---------------------------------------------------------------------
# Vivarium runner
VIVARIUM_TOOL_NAME = "vivarium_module_generator"

VIVARIUM_TOOL_DESCRIPTION = {
    "type": "function",
    "function": {
        "name": "vivarium_module_generator",
        "description": "Convert biological rules into Vivarium-compatible Python module code.",
        "parameters": {
            "type": "object",
            "properties": {
                "cell_type": {"type": "string", "description": "Type of the cell (e.g., T-cell, macrophage)"},
                "rules": {"type": "string", "description": "Rules and interactions governing the cell type"}
            },
            "required": ["cell_type", "rules"]
        }
    }
}

# tool handler for Vivarium
def run_vivarium_module_generator(cell_type: str, rules: str) -> str:
    """
    Generate and simulate a Vivarium module from biological rules for a specific cell type.

    :param cell_type: Type of cell (e.g. T-cell, macrophage)
    :param rules: LLM-generated logic as Python code
    :return: Final simulation state summary
    """
    # Generate full module string with Vivarium Process
    module_code = f"""
from vivarium.core.process import Process

class {cell_type.replace("-", "").replace(" ", "")}Process(Process):
    def __init__(self, config=None):
        super().__init__(config)

    def ports_schema(self):
        return {{
            'internal': {{
                'signal': {{'_default': 0.0}},
                'trigger': {{'_default': False}},
                'time': {{'_default': 0.0}},
                'response': {{'_default': False}}
            }}
        }}

    def next_update(self, timestep, states):
        update = {{'internal': {{}}}}
        # Example rule logic inserted here
        {rules}
        update['internal']['time'] = states['internal']['time'] + timestep
        return update
"""

    with tempfile.NamedTemporaryFile(suffix=".py", mode="w+", delete=False) as temp_file:
        temp_file.write(module_code)
        temp_file_path = temp_file.name

    module_name = os.path.splitext(os.path.basename(temp_file_path))[0]

    try:
        # Import dynamically
        spec = importlib.util.spec_from_file_location(module_name, temp_file_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # Get the process class dynamically
        process_class_name = f"{cell_type.replace('-', '').replace(' ', '')}Process"
        process_class = getattr(mod, process_class_name)
        process = process_class()

        # Build and run simulation
        topology = {'internal': ('internal',)}
        composite = Composite({'processes': {'p': process}, 'topology': {'p': topology}})
        sim_engine = Engine(composite)
        sim_engine.update(0)

        for _ in range(10):  # Run 30 steps
            sim_engine.update(1.0)

        final_state = sim_engine.get_state()
        return f"Vivarium simulation completed for {cell_type}.\n\nFinal state:\n{json.dumps(final_state, indent=2)}"

    except Exception as e:
        return f"Simulation error for {cell_type}: {e}"

    finally:
        os.remove(temp_file_path)

