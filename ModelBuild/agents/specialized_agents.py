# agents/specialized_agents.py

import os
import json
import re
import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

from .base_agents import Agent
from memory.pdf_vector_db import ContentType

logger = logging.getLogger(__name__)

try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI not available")


class PIAgent(Agent):
    """Principal Investigator Agent with task orchestration capabilities."""

    def __init__(self):
        super().__init__(
            title="Principal Investigator Agent",
            expertise="Biological modeling, task orchestration, project management",
            goal="Interpret user prompts, create comprehensive task plans, and coordinate multi-agent workflows",
            role="Project orchestrator and strategic planner",
            model="gpt-4o",
            output_format="""
            Provide a structured task plan in JSON format:
            {
                "project_overview": "Brief description of the parameter extraction task",
                "target_parameter": "Specific parameter to extract",
                "cell_type": "Target cell type (t-cell, tumor, dendritic, lymph_node)",
                "search_strategy": "How to search the literature database",
                "extraction_approach": "Methodology for parameter extraction",
                "validation_criteria": "How to validate the extracted parameter"
            }
            """,
            enable_confidence=True,
            max_retries=2,
            timeout=120
        )

    def route_task(self, prompt: str) -> dict:
        """Route tasks to appropriate agents based on prompt."""
        return {
            "biologist": {"task": "Map biological processes"},
            "parameter": {"task": "Extract relevant parameters"},
            "modelbuilder": {"task": "Generate Vivarium code"},
            "simulation": {"task": "Run simulations"},
        }

    def extract_task_plan(self, response: str) -> Dict[str, Any]:
        """Extract structured task plan from agent response."""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                # Fallback to basic structure parsing
                return self._parse_task_plan_fallback(response)
        except Exception as e:
            logger.error(f"Error extracting task plan: {e}")
            return {"error": str(e), "raw_response": response}

    def _parse_task_plan_fallback(self, response: str) -> Dict[str, Any]:
        """Fallback parsing for task plan."""
        return {
            "project_overview": "Parameter extraction task",
            "target_parameter": "As specified in user request",
            "search_strategy": "Search scientific literature database",
            "extraction_approach": "Extract parameter value with units and citation",
            "validation_criteria": "Check biological plausibility and cite sources"
        }


class BiologistAgent(Agent):
    """Biologist Agent for contextualizing biological parameters."""

    def __init__(self):
        super().__init__(
            title="Biologist Agent",
            expertise="Cell biology, signaling pathways, immune system modeling, parameter contextualization",
            goal="Provide biological context for parameter extraction and validate biological plausibility",
            role="Biological context expert and validator",
            model="gpt-4o",
            output_format="""
            Provide biological context in structured format:
            {
                "parameter_context": {
                    "biological_function": "What this parameter controls biologically",
                    "typical_range": "Expected range of values for this parameter",
                    "measurement_context": "How this parameter is typically measured"
                },
                "validation_criteria": {
                    "units_expected": "Expected units for this parameter",
                    "key_references": "Important papers that measure this parameter"
                }
            }
            """,
            enable_confidence=True
        )

    def map_biology(self, prompt: str) -> dict:
        """Map biological processes from prompt."""
        return {"TCell": {"interacts_with": ["TumorCell"], "actions": ["kill", "secrete_IFNg"]}}


class ParameterExtractorAgent(Agent):
    """Parameter Extractor Agent with advanced search and extraction capabilities."""

    def __init__(self):
        super().__init__(
            title="Parameter Extractor Agent",
            expertise="Scientific literature analysis, quantitative parameter extraction, database search",
            goal="Extract biological parameters from scientific literature with high accuracy",
            role="Scientific literature data extraction specialist",
            model="gpt-4o",
            output_format="""
            Format your parameter extraction as:

            PARAMETER EXTRACTION RESULT:
            Parameter: [exact parameter name]
            Value: [numerical value OR inferred approximation]
            Units: [measurement units or inferred]
            Source: [paper citation or section reference]
            Context: [biological or experimental context where this was estimated]
            Confidence: [0.0-1.0 confidence score]

            SUPPORTING EVIDENCE:
            [Explain how you inferred the value and what clues or figures/tables you used. Acknowledge if it's approximate.]
            """,
            enable_confidence=True,
            dependencies=["BiologistAgent"],
            max_retries=3,
            timeout=180
        )
        self.logger = logging.getLogger(__name__)

    def run(self, conversation):
        """Execute parameter extraction with actual OpenAI API calls."""
        try:
            self.start_execution()

            if not self.client:
                logger.error("OpenAI client not initialized. Please set OPENAI_API_KEY environment variable.")
                error_result = self._create_error_response("OpenAI client not available")
                self.complete_execution(error_result)
                return error_result

            # Make the actual API call to extract parameters
            response = self.client.chat.completions.create(
                model=self.model,
                messages=conversation,
                temperature=0.3,
                max_tokens=1000
            )

            result = response.choices[0].message.content
            logger.info(f"Parameter extraction response length: {len(result)}")

            # Extract confidence if available
            confidence = self._extract_confidence_from_response(result)

            self.complete_execution(result, confidence)
            return result

        except Exception as e:
            error_msg = f"Error in Parameter Extractor Agent: {str(e)}"
            logger.error(error_msg)
            self.error_execution(e)
            return self._create_error_response(error_msg)

    def _create_error_response(self, error_msg: str) -> str:
        """Create a structured error response."""
        return f"""
PARAMETER EXTRACTION RESULT:
Parameter: [ERROR]
Value: [NOT_FOUND]
Units: [UNKNOWN]
Source: [ERROR]
Context: {error_msg}
Confidence: 0.0

SUPPORTING EVIDENCE:
Extraction failed due to technical error.
"""

    def _extract_confidence_from_response(self, response: str) -> Optional[float]:
        """Extract confidence score from response."""
        try:
            confidence_match = re.search(r'Confidence:\s*([0-9]*\.?[0-9]+)', response)
            if confidence_match:
                return float(confidence_match.group(1))
        except Exception:
            pass
        return None

    def search_literature_and_extract(self, db, parameter_name, cell_type, prompt_text, top_k=5):
        """
        Try to extract parameter from literature. If not found, fall back to model's own knowledge.
        """
        from memory.pdf_vector_db import ContentType

        # Step 1: Search vector DB
        try:
            excerpts = db.search_content(
                query=f"{parameter_name} in {cell_type} cells",
                top_k=top_k,
                content_types=[ContentType.PDF_PAPER]
            )
        except Exception as e:
            excerpts = []
            self.logger.warning(f"Vector DB query failed: {e}")

        if excerpts:
            self.logger.info(f"Found {len(excerpts)} unique literature excerpts for {parameter_name}")
            context = "\n\n".join(
                f"Excerpt {i + 1} from {res['metadata'].get('file_name', 'Unknown')}:\n{res['content']}"
                for i, res in enumerate(excerpts)
            )
            user_query = (
                f"Based on the following literature excerpts, extract the value of the parameter '{parameter_name}' "
                f"for {cell_type} cells.\n\n{context}\n\n"
                f"Prompt: {prompt_text}\n\n"
                "Be specific and mention units if available. End with CONFIDENCE: <score>"
            )
        else:
            self.logger.warning(f"No relevant excerpts found in vector DB for {parameter_name}")
            user_query = (
                f"No literature excerpts were found for '{parameter_name}' in {cell_type} cells.\n\n"
                f"Using your own biological knowledge and reasoning, what is the most likely value of this parameter?\n\n"
                f"Prompt: {prompt_text}\n\n"
                "Be specific and mention units if available. End with CONFIDENCE: <score>"
            )

        conversation = [
            self.system_message(),
            {"role": "user", "content": user_query}
        ]

        response = self.run(conversation)
        self.logger.info("Successfully extracted parameter using OpenAI")
        return response

    def _build_extraction_context(self, results: List[Dict], parameter_name: str, cell_type: str) -> str:
        """Build comprehensive context from search results."""
        context = f"\n\nRELEVANT LITERATURE EXCERPTS FOR {parameter_name.upper()} IN {cell_type.upper()} CELLS:\n"
        context += "=" * 80 + "\n"

        for i, result in enumerate(results):
            similarity = result['similarity_score']
            paper_name = result['metadata'].get('file_name', 'Unknown paper')
            content = result['content']

            context += f"\n--- EXCERPT {i + 1} (Relevance: {similarity:.3f}) ---\n"
            context += f"Source: {paper_name}\n"
            context += f"Content: {content[:1500]}...\n"

            if i < len(results) - 1:
                context += "\n" + "-" * 40 + "\n"

        context += "\n" + "=" * 80 + "\n"
        return context

    def _create_extraction_prompt(self, parameter_name: str, cell_type: str,
                                  original_prompt: str, context: str) -> str:
        """Create comprehensive extraction prompt."""
        return f"""
PARAMETER EXTRACTION TASK:

TARGET PARAMETER: {parameter_name}
CELL TYPE: {cell_type}
ORIGINAL QUESTION: {original_prompt}

INSTRUCTIONS:
1. Carefully read through all the literature excerpts provided below
2. Look for numerical values related to "{parameter_name}" for {cell_type} cells
3. Extract the most reliable and well-cited value
4. If multiple values are found, choose the most recent or most cited one
5. Include proper units and source citation
6. Provide a confidence score based on the quality of evidence

SEARCH RESULTS:
{context}

Please extract the parameter following this exact format:

PARAMETER EXTRACTION RESULT:
Parameter: {parameter_name}
Value: [numerical value - if found]
Units: [measurement units - if specified]
Source: [paper name and reference]
Context: [biological context where this was measured]
Confidence: [0.0-1.0 confidence score]

SUPPORTING EVIDENCE:
[Explain why you chose this value, what evidence supports it, and any limitations]

If no reliable numerical value is found, set Value to "NOT_FOUND" and explain why in the supporting evidence.
"""

    def _create_no_data_response(self, parameter_name: str, cell_type: str) -> str:
        """Create response when no data is found."""
        return f"""
PARAMETER EXTRACTION RESULT:
Parameter: {parameter_name}
Value: NOT_FOUND
Units: UNKNOWN
Source: NO_SOURCE_FOUND
Context: No relevant literature found for {parameter_name} in {cell_type} cells
Confidence: 0.0

SUPPORTING EVIDENCE:
No literature excerpts containing information about {parameter_name} for {cell_type} cells were found in the database. This could be because:
1. The parameter is not covered in the available papers
2. Different terminology is used in the literature
3. The database needs more relevant papers for this parameter
"""

    def parse_extracted_parameter(self, response: str) -> Dict[str, Any]:
        """Parse extracted parameter from agent response."""
        try:
            result = {
                "parameter": None,
                "value": None,
                "units": None,
                "source": None,
                "context": None,
                "confidence": None
            }

            # Extract each component using regex
            patterns = {
                "parameter": r"Parameter:\s*(.+?)(?:\n|$)",
                "value": r"Value:\s*([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?|NOT_FOUND)",
                "units": r"Units:\s*(.+?)(?:\n|$)",
                "source": r"Source:\s*(.+?)(?:\n|$)",
                "context": r"Context:\s*(.+?)(?:\n|$)",
                "confidence": r"Confidence:\s*([0-9]*\.?[0-9]+)"
            }

            for key, pattern in patterns.items():
                match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
                if match:
                    if key == "confidence":
                        result[key] = float(match.group(1))
                    elif key == "value" and match.group(1) != "NOT_FOUND":
                        try:
                            result[key] = float(match.group(1))
                        except ValueError:
                            result[key] = match.group(1).strip()
                    else:
                        result[key] = match.group(1).strip()

            return result

        except Exception as e:
            logger.error(f"Error parsing extracted parameter: {e}")
            return {"error": str(e), "raw_response": response}


class ModelBuilderAgent(Agent):
    """Model Builder Agent for generating Vivarium code."""

    def __init__(self):
        super().__init__(
            title="Model Builder Agent",
            expertise="Vivarium simulation framework, Python code generation, biological modeling",
            goal="Convert biological parameters and mappings to executable Vivarium modules",
            role="Simulation code architect and parameter integration specialist",
            model="gpt-4o",
            output_format="""
            Generate complete, functional Vivarium code with:
            1. Proper Vivarium imports and class structure
            2. Process classes with next_update() methods
            3. Correct ports_schema() definitions
            4. Parameter integration with proper units
            5. Comprehensive documentation and comments
            """,
            enable_confidence=True,
            dependencies=["BiologistAgent", "ParameterExtractorAgent"],
            max_retries=2,
            timeout=240
        )

    def build_model(self, bio_map: dict, parameters: dict):
        """Generate Vivarium modules from biological mappings and parameters."""
        print("Generating Vivarium modules...")
        # This would contain the actual model generation logic


class CritiqueAgent(Agent):
    """Critique Agent with validation and feedback capabilities."""

    def __init__(self):
        super().__init__(
            title="Critique Agent",
            expertise="Scientific validation, parameter analysis, quality assurance, biological plausibility",
            goal="Comprehensively evaluate extracted parameters and provide detailed feedback",
            role="Scientific quality assurance specialist and validation expert",
            model="gpt-4o",
            output_format="""
            Provide comprehensive critique in structured format:

            QUALITY ASSESSMENT:
            Completeness: [Complete/Incomplete/Missing Elements]
            Accuracy: [High/Medium/Low confidence in accuracy]
            Citation Quality: [Excellent/Good/Poor/Missing]
            Biological Plausibility: [Highly Plausible/Plausible/Questionable/Implausible]

            RECOMMENDATIONS:
            Action Required: [Accept/Revise/Re-extract/Search Additional Sources]
            Specific Issues: [List of identified problems]

            CONFIDENCE: [0.0-1.0 overall confidence score]
            """,
            enable_confidence=True,
            dependencies=["ParameterExtractorAgent"],
            max_retries=2,
            timeout=120
        )

    def _extract_confidence(self, text: str) -> float:
        """Extract confidence score from agent response."""
        confidence_pattern = r"CONFIDENCE:\s*([0-9]*\.?[0-9]+)"
        match = re.search(confidence_pattern, text, re.IGNORECASE)
        if match:
            return float(match.group(1))
        return None

    def critique(self, content: str) -> str:
        """Provide critique of model content."""
        return f"Looks valid. CONFIDENCE: 0.95"

    def parse_critique_decision(self, response: str) -> Dict[str, Any]:
        """Parse critique decision from agent response."""
        try:
            result = {
                "action_required": None,
                "confidence": None,
                "completeness": None,
                "accuracy": None,
                "biological_plausibility": None,
                "specific_issues": [],
                "recommendations": []
            }

            # Extract action required
            action_match = re.search(r"Action Required:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
            if action_match:
                result["action_required"] = action_match.group(1).strip()

            # Extract confidence
            confidence_match = re.search(r"CONFIDENCE:\s*([0-9]*\.?[0-9]+)", response, re.IGNORECASE)
            if confidence_match:
                result["confidence"] = float(confidence_match.group(1))

            # Extract completeness
            completeness_match = re.search(r"Completeness:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
            if completeness_match:
                result["completeness"] = completeness_match.group(1).strip()

            # Extract accuracy assessment
            accuracy_match = re.search(r"Accuracy:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
            if accuracy_match:
                result["accuracy"] = accuracy_match.group(1).strip()

            # Extract biological plausibility
            plausibility_match = re.search(r"Biological Plausibility:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
            if plausibility_match:
                result["biological_plausibility"] = plausibility_match.group(1).strip()

            return result

        except Exception as e:
            logger.error(f"Error parsing critique decision: {e}")
            return {"error": str(e), "raw_response": response}

    def should_retry_extraction(self, critique_result: Dict[str, Any]) -> bool:
        """Determine if parameter extraction should be retried based on critique."""
        action = critique_result.get("action_required", "").lower()
        confidence = critique_result.get("confidence", 0.0)

        # Retry conditions
        retry_conditions = [
            "revise" in action,
            "re-extract" in action,
            "search additional" in action,
            confidence < 0.6,
            critique_result.get("completeness", "").lower() == "incomplete",
            critique_result.get("accuracy", "").lower() == "low"
        ]

        return any(retry_conditions)

    def critique_parameter_extraction(self, extracted_response: str, parameter_name: str, cell_type: str) -> str:
        """
        Generate a critique of the parameter extraction response.

        Args:
            extracted_response (str): The output from ParameterExtractorAgent.
            parameter_name (str): Name of the parameter extracted.
            cell_type (str): Cell type the parameter pertains to.

        Returns:
            str: Critique response from the agent.
        """
        conversation = [
            self.system_message(),
            {"role": "user", "content": (
                f"Please critique the following extraction for the parameter '{parameter_name}' "
                f"related to {cell_type} cells:\n\n{extracted_response}\n\n"
                "Identify if the source is appropriate, if the value is reasonable, and if the context makes sense. "
                "End your response with a confidence score between 0 and 1 formatted as: CONFIDENCE: <score>"
            )}
        ]

        return self.run(conversation)
