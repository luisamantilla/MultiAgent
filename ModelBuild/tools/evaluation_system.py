# evaluation_system.py
import os
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
from dataclasses import dataclass

from agents.specialized_agents import PIAgent, BiologistAgent, ParameterExtractorAgent, ModelBuilderAgent, CritiqueAgent
from memory.pdf_vector_db import MultiAgentSharedDatabase, ContentType

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    prompt_id: str
    cell_type: str
    parameter_name: str
    ground_truth_value: Any
    extracted_value: Optional[Any]
    agent_response: str
    execution_time: float
    confidence_score: Optional[float]
    iteration_count: int
    success: bool

class ParameterEvaluationSystem:
    def __init__(self, db_path: str = "./multi_agent_db", paper_dir: str = "parameter/my_papers"):
        self.db_path = db_path
        self.paper_dir = paper_dir
        self.output_dir = Path("evaluation_results")
        self.output_dir.mkdir(exist_ok=True)

        self.db = MultiAgentSharedDatabase(db_dir=db_path, collection_name="evaluation_memory")
        self.agents = {
            "pi": PIAgent(),
            "biologist": BiologistAgent(),
            "parameter_extractor": ParameterExtractorAgent(),
            "model_builder": ModelBuilderAgent(),
            "critique": CritiqueAgent()
        }

        self.ground_truth = self._load_ground_truth()
        self.level1_prompts = self._create_level1_prompts()
        self.results: List[EvaluationResult] = []

    def _load_ground_truth(self) -> Dict[str, Any]:
        with open("parameter/ground_truth.json") as f:
            return json.load(f)

    def _create_level1_prompts(self) -> List[Dict[str, Any]]:
        with open("parameter/level1_prompts.json") as f:
            return json.load(f)

    async def run_single_evaluation(self, prompt_data: Dict[str, Any], iteration: int = 0) -> EvaluationResult:
        from datetime import datetime
        start_time = datetime.now()

        prompt_id = prompt_data["id"]
        cell_type = prompt_data["cell_type"]
        parameter = prompt_data["parameter"]
        prompt_text = prompt_data["prompt"]

        extractor_agent = self.agents["parameter_extractor"]
        extraction_result = extractor_agent.search_literature_and_extract(
            db=self.db,
            parameter_name=parameter,
            cell_type=cell_type,
            prompt_text=prompt_text,
            top_k=5
        )
        parsed_result = extractor_agent.parse_extracted_parameter(extraction_result)

        self.db.store_agent_memory(
            agent_name="ParameterExtractorAgent",
            iteration=iteration,
            content=extraction_result,
            content_type=ContentType.EXTRACTED_PARAMETER,
            metadata={
                "prompt_id": prompt_id,
                "cell_type": cell_type,
                "parameter": parameter,
                "parsed_value": parsed_result.get("value"),
                "parsed_confidence": parsed_result.get("confidence")
            }
        )

        extracted_value = parsed_result.get("value")
        confidence_score = parsed_result.get("confidence", 0.0)
        execution_time = (datetime.now() - start_time).total_seconds()

        return EvaluationResult(
            prompt_id=prompt_id,
            cell_type=cell_type,
            parameter_name=parameter,
            ground_truth_value=self.ground_truth[cell_type][parameter],
            extracted_value=extracted_value,
            agent_response=extraction_result,
            execution_time=execution_time,
            confidence_score=confidence_score,
            iteration_count=1,
            success=True
        )

    async def run_full_evaluation(self):
        await self.setup_database()

        for i, prompt in enumerate(self.level1_prompts):
            result = await self.run_single_evaluation(prompt, iteration=i)
            self.results.append(result)

        await self.generate_evaluation_report()

    async def setup_database(self):
        self.db.add_pdf_directory(self.paper_dir)

    async def generate_evaluation_report(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"evaluation_report_{timestamp}.json"

        report = {
            "results": [
                {
                    "prompt_id": r.prompt_id,
                    "cell_type": r.cell_type,
                    "parameter_name": r.parameter_name,
                    "ground_truth_value": r.ground_truth_value,
                    "extracted_value": r.extracted_value,
                    "confidence_score": r.confidence_score,
                    "execution_time": r.execution_time,
                    "agent_response": r.agent_response,
                    "success": r.success
                } for r in self.results
            ]
        }

        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        print(f"Evaluation report saved to: {report_file}")
