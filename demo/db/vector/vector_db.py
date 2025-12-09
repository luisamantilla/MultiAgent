import os
import uuid
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from typing import List
import torch
from transformers import AutoTokenizer, AutoModel

# openai client
load_dotenv()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION_URL = os.getenv("QDRANT_COLLECTION_URL")

QDRANT_CLIENT = QdrantClient(
    url=QDRANT_COLLECTION_URL, 
    api_key=QDRANT_API_KEY,
)

_MODEL_ID = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"

def initialize_client(): 
    QDRANT_CLIENT.create_collection(
        collection_name="cell_rules",
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
    )

_tokenizer = None
_model = None
def _lazy_load():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(_MODEL_ID, use_fast=True)
        _model = AutoModel.from_pretrained(_MODEL_ID, use_safetensors=True)
        _model.eval()

def embed(text: str) -> List[float]:
    _lazy_load()
    with torch.no_grad():
        inputs = _tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        out = _model(**inputs).last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1)
        pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled[0].cpu().tolist()
    
def insert_rule_vector_db(cell_rule: str, embedding: float):
    QDRANT_CLIENT.upsert(
        collection_name="cell_rules",
        points=[
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "cell_rule": {cell_rule},
                }
            )
        ]
    )
