import os
from openai import OpenAI
from qdrant_client import QdrantClient
from demo.db.vector.vector_db import embed
from qdrant_client.models import NamedVector

QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION_URL = os.getenv("QDRANT_COLLECTION_URL")
OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")

QDRANT_CLIENT = QdrantClient(
    url=QDRANT_COLLECTION_URL, 
    api_key=QDRANT_API_KEY,
)
OPEN_AI_CLIENT = OpenAI(api_key=OPEN_AI_API_KEY)

def query_qdrant(text: str, collection_limit=200):
    vector = embed(text)
    hits = QDRANT_CLIENT.search(
        collection_name="pubmed_articles",
        query_vector=NamedVector(name="embedding", vector=vector),
        limit=collection_limit
    )
    
    max_min_point = QDRANT_CLIENT.retrieve(
        collection_name="pubmed_articles",
        ids=[0],
        with_payload=True
    )[0].payload
    
    results = []
    max_score = max_min_point["max_numeric"]
    min_score = max_min_point["min_numeric"]
    range = max_score - min_score
    
    for hit in hits:
        results.append({
            "doi": hit.payload.get("doi"),
            "score": (0.3 * hit.payload.get("numeric_score")/range) + (0.7 * hit.score)
        })
        
    sorted_data = sorted(results, key=lambda d: -d["score"])
    final_data = sorted_data[:20]
    
    articles = set()
    for data in final_data:
        articles.add(data.get("doi"))
    return articles

if __name__ == "__main__":
    print(query_qdrant("What is the growth rate and division of PD1+ cells in tumor microenvironments?"))
    