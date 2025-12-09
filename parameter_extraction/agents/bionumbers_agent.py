"""
Retrieval helper that surfaces the top-k local embedding matches and the top-k
BioNumbers entries for every query. BioNumbers hits are persisted into
merged_parameters.json so the local database grows over time.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List
import json
import logging

from external_sources.bionumbers_client import BioNumbersClient
from utils.vector_store import query_vector_db, upsert_record


MERGED_FILE = Path("parameter_extraction/results/merged_parameters.json")

logger = logging.getLogger(__name__)


class BioNumbersRetrievalAgent:
    def __init__(
        self,
        local_k: int = 5,
        bionumbers_k: int = 3,
        persist_bionumbers: bool = True,
    ):
        self.local_k = local_k
        self.bionumbers_k = bionumbers_k
        self.persist_bionumbers = persist_bionumbers
        self.client = BioNumbersClient()

    def answer_parameter_query(self, query: str) -> Dict[str, List[Dict[str, Any]]]:
        """Return both local vector matches and BioNumbers candidates."""
        query = (query or "").strip()
        local_hits = self.get_local_hits(query, k=self.local_k)
        bionumbers_hits = self.get_bionumbers_hits(
            query, k=self.bionumbers_k, persist=self.persist_bionumbers
        )
        return {"local": local_hits, "bionumbers": bionumbers_hits}

    def get_local_hits(self, query: str, k: int | None = None) -> List[Dict[str, Any]]:
        if not query:
            return []
        hits = query_vector_db(query, k=k or self.local_k)
        formatted = []
        for hit in hits:
            # query_vector_db already returns the data in the correct format
            parameter = hit.get("parameter")
            if not parameter:
                continue
            formatted.append(
                {
                    "score": hit.get("score"),
                    "parameter": parameter,
                    "instances": hit.get("instances", []),
                }
            )
        return formatted

    def get_bionumbers_hits(
        self, query: str, k: int | None = None, persist: bool = True
    ) -> List[Dict[str, Any]]:
        if not query:
            return []
        bnids = self.client.search(query, max_results=k or self.bionumbers_k)
        results: List[Dict[str, Any]] = []
        for bnid in bnids:
            entry = self.client.fetch_entry(bnid)
            if not entry:
                continue
            param_doc = entry.to_parameter_doc()
            if persist:
                self._persist_parameter(param_doc)
                upsert_record(param_doc)
            results.append(
                {
                    "bnid": entry.bnid,
                    "parameter": param_doc["parameter"],
                    "instances": param_doc.get("instances", []),
                }
            )
        return results

    def _persist_parameter(self, param_doc: Dict[str, Any]) -> None:
        """Insert/append a parameter entry into merged_parameters.json."""
        merged = load_merged_file()
        name = param_doc["parameter"]
        new_instances = param_doc.get("instances", [])
        if not new_instances:
            return
        if name not in merged["parameters"]:
            merged["parameters"][name] = {"instances": new_instances}
        else:
            merged_instances = merged["parameters"][name].setdefault("instances", [])
            for inst in new_instances:
                if inst not in merged_instances:
                    merged_instances.append(inst)
        save_merged_file(merged)


def load_merged_file() -> Dict[str, Any]:
    if MERGED_FILE.exists():
        return json.loads(MERGED_FILE.read_text(encoding="utf-8"))
    return {"parameters": {}, "summaries": []}


def save_merged_file(data: Dict[str, Any]) -> None:
    MERGED_FILE.parent.mkdir(parents=True, exist_ok=True)
    MERGED_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
