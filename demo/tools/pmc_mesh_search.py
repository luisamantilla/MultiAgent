import os
import uuid
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue

from demo.db.vector.vector_db import embed
from demo.tools.pubmed_queries import fetch_pmc_xml, get_pmid_to_pmcid, get_pmids
from demo.tools.pubmed_query_agent import get_pmc_queries
from demo.tools.regex import COMPILLED_PATTERNS
from demo.tools.xml_functions import extract_paragraphs_by_sections, extract_tables, extract_text_from_pmc

load_dotenv()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION_URL = os.getenv("QDRANT_COLLECTION_URL")

MAX_PMIDS = 1000
BATCH = 256
QUERY_PARAMETERS = ["activation refractory time", "PD1p death rate", "PD1 death rate", "PD1 IFNg production rate", "PD1p IFNg production rate", "PD1 growth rate", "PD1p growth rate"]

QDRANT_CLIENT = QdrantClient(
    url=QDRANT_COLLECTION_URL, 
    api_key=QDRANT_API_KEY,
)

def _doi_exists(doi: str) -> bool:
    """
    Returns True if a point with this DOI is already in collection
    """
    hits, _ = QDRANT_CLIENT.scroll(
        collection_name="pubmed_articles",
        scroll_filter=Filter(
            must=[FieldCondition(key="doi", match=MatchValue(value=doi))]
        ),
        limit=1,
    )
    return bool(hits)


def numeric_score_calculation(text):
    """
    Calculates numeric score for each article using regex (all regex located under regex.py)
    +1 for each regex matched
    """
    total = 0
    for pat in COMPILLED_PATTERNS:
        matches = pat.findall(text)
        total += len(matches)
        
    return total

def update_min_max_paragraph(new_value: int):
    min_max_point = QDRANT_CLIENT.retrieve(
        collection_name="pubmed_articles",
        ids=[0],
        with_payload=True
    )[0].payload
    
    min_max_point["min_numeric"] = min(min_max_point["min_numeric"], new_value)
    min_max_point["max_numeric"] = max(min_max_point["max_numeric"], new_value)
    
    QDRANT_CLIENT.set_payload(
        collection_name="pubmed_articles",
        payload = {
            "min_numeric": min_max_point["min_numeric"],
            "max_numeric": min_max_point["max_numeric"]
        },
        points=[0]
    )
    
def update_min_max_article(new_value: int):
    min_max_point = QDRANT_CLIENT.retrieve(
        collection_name="pubmed_articles",
        ids=[1],
        with_payload=True
    )[0].payload
    
    min_max_point["min_numeric"] = min(min_max_point["min_numeric"], new_value)
    min_max_point["max_numeric"] = max(min_max_point["max_numeric"], new_value)
    
    QDRANT_CLIENT.set_payload(
        collection_name="pubmed_articles",
        payload = {
            "min_numeric": min_max_point["min_numeric"],
            "max_numeric": min_max_point["max_numeric"]
        },
        points=[1]
    )

def run_pubmed_search(article_count: int, query_parameters=QUERY_PARAMETERS):
    """
    Search through PubMed Central + Elseiver for full articles

    """
    
    # ---- PUBMED SEARCH BEGIN ---- #
    
    # queries from parameters
    queries = []
    for parameter in query_parameters:
        queries.extend(get_pmc_queries(parameter, "T-cell").split(","))
    
    temp_article_count = article_count
    for query in queries:
        print("QUERY: ", query)        
        
        # query pubmed
        pmids = get_pmids(query, MAX_PMIDS)
        article_count = temp_article_count

        for pmid in pmids:
            if(article_count <= 0):
                break
            # convert pmid --> pmcid
            pmcid = get_pmid_to_pmcid(pmid) if get_pmid_to_pmcid(pmid) is not None else None
            
            if pmcid is None:
                print("PMCID is None")
                continue
            
            if _doi_exists(pmcid):
                print("DOI/PMCID exists already")
                continue
            
            try:
                print(pmcid)
                
                # fetch xml tree and doi from pmc api
                tree, doi = fetch_pmc_xml(pmcid)
                
                # remove metadata
                text = extract_text_from_pmc(tree)
                
                numeric_score_total = numeric_score_calculation(text)
                print("total numeric score:", numeric_score_total)
                
                # update global min max
                update_min_max_article(numeric_score_total)
                    
                # abstract, result_paragraphs, conclusion_paragraphs = extract_full_text(tree)
                paragraphs = extract_paragraphs_by_sections(tree, wanted={"abstract","methods","results","discussion"})
                captions, tables = extract_tables(tree)
                
                for idx, caption in enumerate(captions):
                    caption_embedding = embed(caption)
                    
                    QDRANT_CLIENT.upsert(
                        collection_name="pubmed_tables",
                        points=[
                            PointStruct(
                                id=str(uuid.uuid4()),
                                vector={
                                    "embedding": caption_embedding
                                },
                                payload={
                                    "doi": doi,
                                    "table": tables[idx],
                                    "table_idx": idx,
                                }
                            )
                        ]
                    )
                
                buf = []
                # -- Embedding and storage of abstracts, conclusions, and results -- #
                for key, value in paragraphs.items():
                    for idx, p in enumerate(value):
                        emb = embed(p)

                        score = numeric_score_calculation(p)
                        update_min_max_paragraph(score)

                        buf.append(
                            PointStruct(
                                id=str(uuid.uuid4()),
                                vector={"embedding": emb},
                                payload={
                                    "doi": doi,
                                    "section": key,
                                    "p": p,
                                    "para_idx": idx,
                                    "numeric_score": score,
                                }
                            )
                        )

                        if len(buf) >= BATCH:
                            QDRANT_CLIENT.upsert(
                                collection_name="pubmed_articles",
                                points=buf,
                                wait=True
                            )
                            buf.clear()

                # flush remainder
                if buf:
                    QDRANT_CLIENT.upsert(
                        collection_name="pubmed_articles",
                        points=buf,
                        wait=True
                    )
                
                # ensure article_count is lowered
                article_count -= 1
                print(article_count)
            except Exception as e:
                print(f"Failed {pmcid}: {e}")
            
    # ---- PUBMED SEARCH END ---- #
    
    # ---- ELSEIVER SEARCH BEGIN ---- #
    
    # ---- PUBMED SEARCH END ---- #
      
if __name__ == "__main__":     
    run_pubmed_search(150)
