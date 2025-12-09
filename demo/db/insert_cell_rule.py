import os
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from demo.db.traditional.insert_traditional_db import insert_rule_traditional
from demo.db.vector.vector_db import embed, insert_rule_vector_db, search_db

# qdrant client
qdrant_client = QdrantClient(host="localhost", port=6333)

# openai client
load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def insert_cell_rule(rule_to_insert: str, reference: str, weight: float, cell_type_id: int):
    # embedding = embed(rule_to_insert)

    # Search for similar rules
    # search_result = search_db(embedding, 0.85)

    # if search_result:
    #     most_similar = search_result[0]
    #     raise ValueError(f"Similar cell rule found with score {most_similar.score:.3f}: {most_similar.payload['rule']}")

    # Insert into vector DB and relational DB
    # insert_rule_vector_db(rule_to_insert, embedding)
    insert_rule_traditional(rule_to_insert, reference, weight, cell_type_id)
    

    