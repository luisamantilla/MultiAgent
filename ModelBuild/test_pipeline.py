# test_pipeline.py

import os
import asyncio
import json
import re
from pathlib import Path
from datetime import datetime
import logging

from memory.pdf_vector_db import MultiAgentSharedDatabase, ContentType
from agents.specialized_agents import ParameterExtractorAgent, CritiqueAgent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
DB_DIR = "multi_agent_db"
COLLECTION_NAME = "evaluation_memory"
OUTPUT_PATH = "evaluation_results/test_level1_output_2.json"
TOP_K = 5


def load_prompts():
    """Load test prompts from JSON file."""
    prompt_file = Path("tumor_tcell/level1_prompts_2.json")
    if not prompt_file.exists():
        logger.error(f"Prompt file not found: {prompt_file}")
        # Create a minimal test set if file doesn't exist
        return [
            {"id": "Q1", "parameter": "time_step", "cell_type": "t-cell",
             "prompt": "What is the timestep duration used for T cell simulations in seconds?"},
            {"id": "Q2", "parameter": "t_cell_diameter", "cell_type": "t-cell",
             "prompt": "What is the diameter of a T cell in micrometers?"}
        ]

    with open(prompt_file) as f:
        return json.load(f)


def extract_confidence_score(text: str) -> float:
    """Extract confidence score from agent response."""
    patterns = [
        r"CONFIDENCE[:：]?\s*([0-1](?:\.\d+)?)",
        r"Confidence[:：]?\s*([0-1](?:\.\d+)?)",
        r"confidence[:：]?\s*([0-1](?:\.\d+)?)"
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue

    return 0.5  # Default confidence if not found


class FullTextPaperRetriever:
    """Retrieves full text of papers rather than just chunks."""

    def __init__(self, db: MultiAgentSharedDatabase):
        self.db = db

    def get_full_paper_texts(self, query: str, top_k: int = 2) -> list:
        """Get full text of top-k most relevant papers."""
        try:
            # First, get relevant chunks with paper metadata
            search_results = self.db.search_content(
                query=query,
                top_k=top_k * 3,  # Get more chunks to find top papers
                content_types=[ContentType.PDF_PAPER]
            )

            if not search_results:
                logger.warning(f"No search results found for query: {query}")
                return []

            # Group chunks by paper and calculate average relevance
            papers_relevance = {}
            for result in search_results:
                file_name = result['metadata'].get('file_name', 'Unknown')
                file_path = result['metadata'].get('file_path', '')

                if file_name not in papers_relevance:
                    papers_relevance[file_name] = {
                        'file_path': file_path,
                        'file_name': file_name,
                        'relevance_scores': [],
                        'chunks': []
                    }

                papers_relevance[file_name]['relevance_scores'].append(result['similarity_score'])
                papers_relevance[file_name]['chunks'].append(result['content'])

            # Calculate average relevance and sort papers
            for paper_data in papers_relevance.values():
                scores = paper_data['relevance_scores']
                paper_data['avg_relevance'] = sum(scores) / len(scores) if scores else 0.0

            # Sort by average relevance and take top_k
            sorted_papers = sorted(
                papers_relevance.values(),
                key=lambda x: x['avg_relevance'],
                reverse=True
            )[:top_k]

            # Get full text for each top paper
            full_papers = []
            for paper_data in sorted_papers:
                file_path = paper_data['file_path']
                if file_path and os.path.exists(file_path):
                    full_text = self._extract_full_pdf_text(file_path)
                    if full_text:
                        full_papers.append({
                            'file_name': paper_data['file_name'],
                            'file_path': file_path,
                            'content': full_text,
                            'avg_relevance': paper_data['avg_relevance'],
                            'metadata': {'file_name': paper_data['file_name']}
                        })
                    else:
                        logger.warning(f"Could not extract text from {file_path}")
                else:
                    # Fallback to concatenated chunks if file not accessible
                    full_papers.append({
                        'file_name': paper_data['file_name'],
                        'file_path': file_path,
                        'content': '\n\n'.join(paper_data['chunks']),
                        'avg_relevance': paper_data['avg_relevance'],
                        'metadata': {'file_name': paper_data['file_name']}
                    })

            logger.info(f"Retrieved {len(full_papers)} full papers for query: {query}")
            for i, paper in enumerate(full_papers):
                logger.info(f"  {i + 1}. {paper['file_name']} (relevance: {paper['avg_relevance']:.3f})")

            return full_papers

        except Exception as e:
            logger.error(f"Error retrieving full paper texts: {e}")
            return []

    def _extract_full_pdf_text(self, pdf_path: str) -> str:
        """Extract full text from PDF file."""
        try:
            return self.db._extract_text_from_pdf(pdf_path)
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return ""


class EnhancedParameterExtractor:
    """Enhanced parameter extractor that handles inference from context and figures."""

    def __init__(self):
        self.base_agent = ParameterExtractorAgent()

    def create_enhanced_prompt(self, parameter_name: str, cell_type: str,
                               original_prompt: str, full_papers: list) -> str:
        """Create enhanced prompt for parameter extraction with full paper context."""

        papers_context = ""
        for i, paper in enumerate(full_papers):
            papers_context += f"\n{'=' * 80}\n"
            papers_context += f"PAPER {i + 1}: {paper['file_name']}\n"
            papers_context += f"RELEVANCE SCORE: {paper['avg_relevance']:.3f}\n"
            papers_context += f"{'=' * 80}\n"
            papers_context += f"{paper['content'][:8000]}..."  # Limit to avoid token limits
            if len(paper['content']) > 8000:
                papers_context += f"\n\n[Content truncated - full paper has {len(paper['content'])} characters]"
            papers_context += f"\n{'=' * 80}\n"

        enhanced_prompt = f"""
ADVANCED PARAMETER EXTRACTION TASK

TARGET PARAMETER: {parameter_name}
CELL TYPE: {cell_type}
ORIGINAL QUESTION: {original_prompt}

EXTRACTION INSTRUCTIONS:
1. Carefully analyze the full text of the provided papers
2. Look for explicit numeric values for "{parameter_name}" in {cell_type} cells
3. If no explicit value is found, infer from:
   - Mathematical models and equations
   - Figure data and graphs
   - Comparative statements (e.g., "10-fold higher than...")
   - Related parameters that can be converted
   - Typical ranges mentioned in the literature
4. Consider biological context and model assumptions
5. If multiple values are found, choose the most reliable/recent one
6. Provide reasoning for your extraction or inference

IMPORTANT GUIDELINES:
- Parameters may not be explicitly stated but can be inferred from context
- Look for symbolic representations (e.g., ρIL, ρIFN) that might correspond to rates
- Consider model parameters that might represent the target parameter
- Check figures, tables, and supplementary material descriptions
- If the paper describes a model, look for parameter values in the methods or results
- Consider physiological ranges and biological plausibility

PAPER SOURCES:
{papers_context}

RESPONSE FORMAT:
PARAMETER EXTRACTION RESULT:
Parameter: {parameter_name}
Value: [numerical value or "INFERRED_FROM_CONTEXT" or "NOT_FOUND"]
Units: [measurement units if available]
Source: [specific paper and section/figure if found]
Context: [detailed explanation of how value was found or inferred]
Confidence: [0.0-1.0 confidence score]

SUPPORTING EVIDENCE:
[Detailed explanation including:
- Exact quotes or references if explicit value found
- Inference methodology if value was derived
- Biological rationale for the extracted/inferred value
- Any limitations or assumptions made]

If no reliable value can be extracted or inferred, explain why and what additional information would be needed.
"""
        return enhanced_prompt


async def run_streamlined_test_pipeline():
    """Run streamlined test pipeline with ParameterAgent and CritiqueAgent only."""
    logger.info("Starting streamlined parameter extraction test pipeline...")

    # Load prompts
    prompts = load_prompts()
    logger.info(f"Loaded {len(prompts)} test prompts")

    # Initialize database
    logger.info("Initializing vector database...")
    db = MultiAgentSharedDatabase(
        db_dir=DB_DIR,
        collection_name=COLLECTION_NAME
    )

    # Check database status
    stats = db.get_database_stats()
    logger.info(f"Database contains {stats.get('total_entries', 0)} entries, "
                f"{stats.get('papers', {}).get('total', 0)} papers")

    if stats.get('papers', {}).get('total', 0) == 0:
        logger.warning("No papers found in database. Adding papers...")
        paper_dir = Path("parameter/my_papers")
        if paper_dir.exists():
            db.add_pdf_directory(str(paper_dir))
            stats = db.get_database_stats()
            logger.info(f"After adding papers: {stats.get('papers', {}).get('total', 0)} papers")

    # Initialize components
    logger.info("Initializing agents...")
    paper_retriever = FullTextPaperRetriever(db)
    enhanced_extractor = EnhancedParameterExtractor()
    critique_agent = CritiqueAgent()

    # Process each prompt
    all_results = []

    for i, prompt_data in enumerate(prompts):
        prompt_id = prompt_data["id"]
        cell_type = prompt_data["cell_type"]
        parameter_name = prompt_data["parameter"]
        prompt = prompt_data["prompt"]

        logger.info(f"\n--- Processing Prompt {i + 1}/{len(prompts)}: {prompt_id} ---")
        logger.info(f"Parameter: {parameter_name} | Cell Type: {cell_type}")

        try:
            # Step 1: Retrieve top 2 relevant papers with full text
            logger.info("Retrieving relevant papers...")
            search_query = f"{parameter_name} {cell_type} cells biological parameter"
            full_papers = paper_retriever.get_full_paper_texts(search_query, top_k=TOP_K)

            if not full_papers:
                logger.warning(f"No relevant papers found for {prompt_id}")
                continue

            top_titles = [paper['file_name'] for paper in full_papers]
            logger.info(f"Top {len(top_titles)} papers: {', '.join(top_titles)}")

            # Step 2: Enhanced parameter extraction
            logger.info("Running enhanced parameter extraction...")
            enhanced_prompt = enhanced_extractor.create_enhanced_prompt(
                parameter_name, cell_type, prompt, full_papers
            )

            conversation = [
                enhanced_extractor.base_agent.system_message(),
                {"role": "user", "content": enhanced_prompt}
            ]

            raw_extraction = enhanced_extractor.base_agent.run(conversation)

            # Parse extraction result
            parsed_result = enhanced_extractor.base_agent.parse_extracted_parameter(raw_extraction)
            confidence_score = extract_confidence_score(raw_extraction)

            logger.info(f"Extraction completed - Value: {parsed_result.get('value')}, "
                        f"Confidence: {confidence_score}")

            # Store parameter extraction in database
            db.store_agent_memory(
                "ParameterExtractorAgent",
                iteration=0,
                content=raw_extraction,
                content_type=ContentType.EXTRACTED_PARAMETER,
                metadata={
                    "prompt_id": prompt_id,
                    "parameter_name": parameter_name,
                    "cell_type": cell_type,
                    "parsed_value": parsed_result.get("value"),
                    "parsed_confidence": confidence_score,
                    "top_papers": top_titles
                }
            )

            # Step 3: Critique agent review
            logger.info("Running critique agent...")
            critique_prompt = f"""
Review the following parameter extraction result:

ORIGINAL TASK: Extract "{parameter_name}" for {cell_type} cells
ORIGINAL PROMPT: {prompt}
TOP PAPERS ANALYZED: {', '.join(top_titles)}

EXTRACTION RESULT:
{raw_extraction}

EVALUATION CRITERIA:
1. Completeness: Is the extraction complete with value, units, and source?
2. Accuracy: Does the extracted value seem biologically plausible?
3. Methodology: Was the extraction/inference methodology sound?
4. Citation Quality: Is the source properly cited?
5. Confidence Assessment: Is the confidence score appropriate?

Provide detailed feedback on the quality of this parameter extraction.
"""

            critique_conversation = [
                critique_agent.system_message(),
                {"role": "user", "content": critique_prompt}
            ]

            critique_output = critique_agent.run(critique_conversation)
            critique_confidence = extract_confidence_score(critique_output)

            logger.info(f"Critique completed - Confidence: {critique_confidence}")

            # Store critique in database
            db.store_agent_memory(
                "CritiqueAgent",
                iteration=0,
                content=critique_output,
                content_type=ContentType.CRITIQUE_SUMMARY,
                metadata={
                    "prompt_id": prompt_id,
                    "parameter_name": parameter_name,
                    "cell_type": cell_type,
                    "critique_confidence": critique_confidence
                }
            )

            # Compile result
            result = {
                "prompt_id": prompt_id,
                "timestamp": datetime.now().isoformat(),
                "prompt": prompt,
                "parameter_name": parameter_name,
                "cell_type": cell_type,
                "top_papers": {
                    "titles": top_titles,
                    "relevance_scores": [paper['avg_relevance'] for paper in full_papers]
                },
                "parameter_extraction": {
                    "raw_output": raw_extraction,
                    "parsed_value": parsed_result.get("value"),
                    "parsed_units": parsed_result.get("units"),
                    "parsed_source": parsed_result.get("source"),
                    "parsed_context": parsed_result.get("context"),
                    "confidence_score": confidence_score
                },
                "critique_analysis": {
                    "raw_output": critique_output,
                    "confidence_score": critique_confidence
                },
                "processing_metadata": {
                    "papers_processed": len(full_papers),
                    "total_paper_characters": sum(len(p['content']) for p in full_papers),
                    "extraction_method": "full_paper_analysis"
                }
            }

            all_results.append(result)
            logger.info(f"Successfully processed {prompt_id}")

        except Exception as e:
            logger.error(f"Error processing {prompt_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())

            # Add error result
            error_result = {
                "prompt_id": prompt_id,
                "timestamp": datetime.now().isoformat(),
                "prompt": prompt,
                "parameter_name": parameter_name,
                "cell_type": cell_type,
                "error": str(e),
                "top_papers": {"titles": [], "relevance_scores": []},
                "parameter_extraction": {"raw_output": f"ERROR: {str(e)}"},
                "critique_analysis": {"raw_output": f"ERROR: Could not critique due to extraction failure"}
            }
            all_results.append(error_result)

    # Save results
    logger.info("Saving results...")
    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    final_output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "total_prompts_processed": len(all_results),
            "successful_extractions": len([r for r in all_results if "error" not in r]),
            "database_stats": stats,
            "pipeline_version": "streamlined_v1.0"
        },
        "results": all_results
    }

    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    logger.info(f"Test pipeline completed successfully!")
    logger.info(f"Results saved to: {output_path}")
    logger.info(f"Processed {len(all_results)} prompts with "
                f"{len([r for r in all_results if 'error' not in r])} successful extractions")

    # Print summary
    print("\n" + "=" * 80)
    print("STREAMLINED TEST PIPELINE SUMMARY")
    print("=" * 80)
    for result in all_results:
        status = "ERROR" if "error" in result else "SUCCESS"
        value = result.get("parameter_extraction", {}).get("parsed_value", "N/A")
        print(f"{result['prompt_id']}: {status} - Value: {value}")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(run_streamlined_test_pipeline())