import os

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

from dotenv import load_dotenv

load_dotenv()

QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION_URL = os.getenv("QDRANT_COLLECTION_URL")

QDRANT_CLIENT = QdrantClient(
    url=QDRANT_COLLECTION_URL, 
    api_key=QDRANT_API_KEY,
)

VECTOR_SIZE = 768

def create_collections():
    if QDRANT_CLIENT.collection_exists("pubmed_articles"):
        QDRANT_CLIENT.delete_collection("pubmed_articles")
        
    if QDRANT_CLIENT.collection_exists("pubmed_tables"):
        QDRANT_CLIENT.delete_collection("pubmed_tables")

    # Create with named vector 'embedding'
    QDRANT_CLIENT.create_collection(
        collection_name="pubmed_articles",
        vectors_config={
            "embedding": VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE
            )
        }
    )

    # Create payload index on 'pmcid'
    QDRANT_CLIENT.create_payload_index(
        collection_name="pubmed_articles",
        field_name="doi",
        field_schema="keyword"
    )
    
    # Create with named vector 'embedding'
    QDRANT_CLIENT.create_collection(
        collection_name="pubmed_tables",
        vectors_config={
            "embedding": VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE
            )
        }
    )

    # Create payload index on 'pmcid'
    QDRANT_CLIENT.create_payload_index(
        collection_name="pubmed_tables",
        field_name="doi",
        field_schema="keyword"
    )
        
    # paragraph numeric global values 
    QDRANT_CLIENT.upsert(
        collection_name="pubmed_articles",
        points=[
            PointStruct(
                id=0,
                vector={"embedding": VECTOR_SIZE * [0.0]},
                payload={
                    "min_numeric": 1e9,
                    "max_numeric": -1e9
                }
            )
        ],
    )
    
    # articles numeric global values
    QDRANT_CLIENT.upsert(
        collection_name="pubmed_articles",
        points=[
            PointStruct(
                id=1,
                vector={"embedding": VECTOR_SIZE * [0.0]},
                payload={
                    "min_numeric": 1e9,
                    "max_numeric": -1e9
                }
            )
        ],
    )

create_collections()