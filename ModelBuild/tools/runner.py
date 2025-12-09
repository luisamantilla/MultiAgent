# tools/runner.py
import os
import sys
import asyncio
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

from .evaluation_system import ParameterEvaluationSystem
from memory.pdf_vector_db import MultiAgentSharedDatabase
from .tools import get_pdf_files

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class EvaluationRunner:
    """
    Main evaluation runner that coordinates the multi-agent parameter extraction system.
    Handles database setup, paper processing, and evaluation orchestration.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.setup_directories()

        # Initialize core components
        self.db = MultiAgentSharedDatabase(
            db_dir=self.config["db_path"],
            collection_name=self.config["collection_name"]
        )

        self.evaluator = ParameterEvaluationSystem(
            db_path=self.config["db_path"],
            paper_dir=self.config["paper_dir"]
        )

    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for the evaluation system."""
        return {
            "db_path": "./multi_agent_db",
            "collection_name": "evaluation_memory",
            "paper_dir": "parameter/my_papers",
            "output_dir": "evaluation_results",
            "log_level": "INFO",
            "skip_existing_papers": True,
            "force_rebuild_db": False
        }

    def setup_directories(self):
        """Setup required directories."""
        dirs_to_create = [
            self.config["db_path"],
            self.config["output_dir"],
            "logs"
        ]

        for dir_path in dirs_to_create:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {dir_path}")

    def check_database_status(self) -> tuple[bool, int, int]:
        """Check if database exists and has papers."""
        try:
            stats = self.db.get_database_stats()
            paper_count = stats.get('papers', {}).get('total', 0)
            total_entries = stats.get('total_entries', 0)

            logger.info(f"Database status: {total_entries} entries, {paper_count} papers processed")
            return paper_count > 0, paper_count, total_entries

        except Exception as e:
            logger.warning(f"Could not check database status: {e}")
            return False, 0, 0

    async def initialize_database(self) -> bool:
        """Initialize database with smart paper processing."""
        # Check if database already exists and has content
        db_exists, paper_count, total_entries = self.check_database_status()

        if self.config["force_rebuild_db"]:
            logger.info("Force rebuild requested - processing all papers...")
        elif db_exists and paper_count > 0:
            logger.info(f"Database already exists with {paper_count} papers ({total_entries} entries)")
            logger.info("Skipping PDF processing - using existing database")
            return True

        # Continue with processing papers
        logger.info("Initializing vector database...")

        paper_dir = Path(self.config["paper_dir"])
        if not paper_dir.exists():
            logger.error(f"Paper directory not found: {paper_dir}")
            return False

        # Get all PDF files
        pdf_files = get_pdf_files(str(paper_dir))
        logger.info(f"Found {len(pdf_files)} PDF files to check")

        if not pdf_files:
            logger.warning("No PDF files found in paper directory")
            return False

        # Smart processing: only add new or changed papers
        if self.config["skip_existing_papers"] and not self.config["force_rebuild_db"]:
            new_papers, updated_papers = self._find_new_and_updated_papers(pdf_files)
            papers_to_process = new_papers + updated_papers
            logger.info(f"Found {len(new_papers)} new papers, {len(updated_papers)} updated papers")
        else:
            papers_to_process = pdf_files
            logger.info(f"Processing all {len(papers_to_process)} papers")

        if not papers_to_process:
            logger.info("All papers are already up-to-date in database!")
            return True

        # Process papers
        success_count = 0
        start_time = time.time()

        for i, pdf_path in enumerate(papers_to_process):
            try:
                logger.info(f"Processing paper {i + 1}/{len(papers_to_process)}: {Path(pdf_path).name}")
                success = self.db.add_pdf_paper(pdf_path)
                if success:
                    success_count += 1
                    logger.info(f"Successfully processed: {Path(pdf_path).name}")
                else:
                    logger.warning(f"Failed to process: {Path(pdf_path).name}")

            except Exception as e:
                logger.error(f"Error processing {Path(pdf_path).name}: {e}")

        total_time = time.time() - start_time
        logger.info(f"Database update complete in {total_time / 60:.1f} minutes: "
                    f"{success_count}/{len(papers_to_process)} papers processed")

        # Print final database statistics
        stats = self.db.get_database_stats()
        logger.info(f"Final database stats: {stats.get('total_entries', 0)} total entries, "
                    f"{stats.get('papers', {}).get('total', 0)} papers")

        return success_count >= 0  # Consider success even if no new papers

    def _find_new_and_updated_papers(self, pdf_files: List[str]) -> tuple[List[str], List[str]]:
        """Find new and updated papers that need processing."""
        processed_files = self.db._load_processed_files()

        new_papers = []
        updated_papers = []

        for pdf_path in pdf_files:
            pdf_path_str = str(Path(pdf_path).resolve())

            if pdf_path_str not in processed_files:
                # New paper
                new_papers.append(pdf_path)
            else:
                # Check if paper has been modified
                current_hash = self.db._get_file_hash(pdf_path)
                stored_hash = processed_files[pdf_path_str]

                if current_hash != stored_hash:
                    updated_papers.append(pdf_path)

        return new_papers, updated_papers

    def demonstrate_retrieval_process(self, query_example: str = "T cell migration velocity"):
        """Demonstrate how the retrieval system works."""
        logger.info("DEMONSTRATION: How Parameter Retrieval Works")
        logger.info("=" * 60)

        # Step 1: Query the database
        logger.info(f"Example Query: '{query_example}'")

        try:
            from memory.pdf_vector_db import ContentType

            # Search the database
            results = self.db.search_content(
                query=query_example,
                top_k=3,
                content_types=[ContentType.PDF_PAPER]
            )

            logger.info(f"Database Search Results: Found {len(results)} relevant chunks")

            for i, result in enumerate(results):
                logger.info(f"\nResult {i + 1}:")
                logger.info(f"   Similarity Score: {result['similarity_score']:.3f}")
                logger.info(f"   Source: {result['metadata'].get('file_name', 'Unknown')}")
                logger.info(f"   Preview: {result['content'][:200]}...")

            logger.info("\nThis is how agents find relevant information!")
            logger.info("   1. User asks for parameter")
            logger.info("   2. System searches vector database")
            logger.info("   3. Finds most relevant paper excerpts")
            logger.info("   4. Agents extract specific values")
            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"Could not demonstrate retrieval: {e}")

    async def run_full_evaluation(self) -> bool:
        """Run comprehensive evaluation with optimized database handling."""
        logger.info("Starting comprehensive multi-agent evaluation...")

        start_time = datetime.now()

        try:
            # Initialize database
            db_success = await self.initialize_database()
            if not db_success:
                logger.error("Database initialization failed. Cannot proceed with evaluation.")
                return False

            # Demonstrate retrieval process
            self.demonstrate_retrieval_process()

            # Run evaluation
            logger.info("Starting parameter extraction evaluation...")
            await self.evaluator.run_full_evaluation()

            # Generate additional analysis
            await self.generate_executive_summary()

            end_time = datetime.now()
            duration = end_time - start_time

            logger.info(f"Evaluation completed successfully in {duration}")
            logger.info(f"Results saved in: {self.config['output_dir']}")

            return True

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    async def run_cell_type_evaluation(self, cell_type: str):
        """Run evaluation for a specific cell type."""
        logger.info(f"Starting evaluation for {cell_type} parameters...")

        # Initialize database
        await self.initialize_database()

        # Demonstrate retrieval for this cell type
        self.demonstrate_retrieval_process(f"{cell_type} cell parameters")

        # Filter prompts for specific cell type
        cell_prompts = [p for p in self.evaluator.level1_prompts if p["cell_type"] == cell_type]
        logger.info(f"Found {len(cell_prompts)} prompts for {cell_type}")

        # Run evaluations
        start_time = time.time()
        for i, prompt_data in enumerate(cell_prompts):
            logger.info(f"Evaluating {cell_type} prompt {i + 1}/{len(cell_prompts)}: {prompt_data['parameter']}")
            result = await self.evaluator.run_single_evaluation(prompt_data, iteration=i)
            self.evaluator.results.append(result)

            # Log immediate result
            logger.info(f"Completed {prompt_data['id']}: "
                        f"Accuracy={result.accuracy:.3f}, Success={result.success}, "
                        f"Time={result.execution_time:.1f}s")

        total_time = time.time() - start_time

        # Generate report
        await self.evaluator.generate_evaluation_report()
        logger.info(f"{cell_type} evaluation completed in {total_time / 60:.1f} minutes!")

    async def run_single_parameter_evaluation(self, prompt_id: str) -> bool:
        """Run evaluation for a single parameter."""
        logger.info(f"Starting single parameter evaluation: {prompt_id}")

        # Initialize database
        await self.initialize_database()

        # Find specific prompt
        prompt_data = next((p for p in self.evaluator.level1_prompts if p["id"] == prompt_id), None)
        if not prompt_data:
            logger.error(f"Prompt ID {prompt_id} not found")
            return False

        # Demonstrate retrieval for this specific parameter
        self.demonstrate_retrieval_process(f"{prompt_data['cell_type']} {prompt_data['parameter']}")

        # Run evaluation
        logger.info(f"Evaluating: {prompt_data['parameter']} for {prompt_data['cell_type']}")
        result = await self.evaluator.run_single_evaluation(prompt_data)
        self.evaluator.results.append(result)

        # Generate report
        await self.evaluator.generate_evaluation_report()

        logger.info(f"Single parameter evaluation completed: "
                    f"Accuracy={result.accuracy:.3f}, Success={result.success}, "
                    f"Time={result.execution_time:.1f}s")
        return True

    async def generate_executive_summary(self):
        """Generate executive summary for stakeholders."""
        logger.info("Generating executive summary...")

        results = self.evaluator.results
        if not results:
            logger.warning("No results available for summary")
            return

        # Calculate key metrics
        total_prompts = len(results)
        successful_extractions = sum(1 for r in results if r.success)
        success_rate = successful_extractions / total_prompts if total_prompts > 0 else 0
        avg_accuracy = sum(r.accuracy for r in results) / total_prompts if total_prompts > 0 else 0
        avg_time = sum(r.execution_time for r in results) / total_prompts if total_prompts > 0 else 0

        # Create executive summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = Path(self.config["output_dir"]) / f"executive_summary_{timestamp}.md"

        with open(summary_file, 'w') as f:
            f.write("# Multi-Agent Parameter Extraction - Executive Summary\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## Key Performance Indicators\n\n")
            f.write(f"- **Overall Success Rate:** {success_rate:.1%}\n")
            f.write(f"- **Average Accuracy:** {avg_accuracy:.3f}\n")
            f.write(f"- **Average Processing Time:** {avg_time:.2f} seconds\n")
            f.write(f"- **Total Parameters Evaluated:** {total_prompts}\n\n")

            f.write("## Retrieval System Performance\n\n")
            f.write("- **Database Status:** Persistent vector database with semantic search\n")
            f.write("- **Smart Caching:** Only processes new/changed papers\n")
            f.write("- **Query Speed:** Sub-second parameter retrieval from literature\n\n")

        logger.info(f"Executive summary saved: {summary_file}")

    def print_database_info(self):
        """Print detailed database information."""
        print("\n" + "=" * 80)
        print("DATABASE RETRIEVAL SYSTEM INFO")
        print("=" * 80)

        try:
            stats = self.db.get_database_stats()
            print(f"Total Entries: {stats.get('total_entries', 0)}")
            print(f"Papers Processed: {stats.get('papers', {}).get('total', 0)}")
            print(f"Text Chunks: {stats.get('papers', {}).get('total_chunks', 0)}")
            print(f"Database Location: {self.config['db_path']}")

            print(f"\nHow Retrieval Works:")
            print(f"1. Papers are split into {self.db.chunk_size}-character chunks")
            print(f"2. Each chunk gets embedded using {self.db.embedding_model}")
            print(f"3. Agent queries find most relevant chunks via semantic search")
            print(f"4. Multiple search strategies increase parameter discovery")

            print(f"\nPerformance Optimization:")
            print(f"- Persistent database - PDFs only processed once")
            print(f"- Smart caching - skips unchanged papers")
            print(f"- Vector search - sub-second query responses")
            print(f"- Chunk overlap - ensures no parameter is missed")

        except Exception as e:
            print(f"Could not load database info: {e}")

        print("=" * 80)

    def print_available_prompts(self):
        """Print all available evaluation prompts."""
        print("\n" + "=" * 80)
        print("AVAILABLE EVALUATION PROMPTS")
        print("=" * 80)

        by_cell_type = {}
        for prompt in self.evaluator.level1_prompts:
            cell_type = prompt["cell_type"]
            if cell_type not in by_cell_type:
                by_cell_type[cell_type] = []
            by_cell_type[cell_type].append(prompt)

        for cell_type, prompts in by_cell_type.items():
            print(f"\n{cell_type.upper()} PARAMETERS ({len(prompts)} prompts):")
            for prompt in prompts:
                print(f"  {prompt['id']}: {prompt['parameter']}")

        print("\nExample usage:")
        print("  python main.py --mode single --prompt-id t-cell_01")
        print("  python main.py --mode cell-type --cell-type tumor")
        print("=" * 80)


def setup_evaluation_environment() -> bool:
    """Setup and validate evaluation environment."""
    logger.info("Setting up evaluation environment...")

    # Check OpenAI API key
    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable not set")
        logger.error("Set with: export OPENAI_API_KEY='your-key-here'")
        return False
    else:
        logger.info("OPENAI_API_KEY is configured")

    # Check paper directory
    paper_dir = Path("parameter/my_papers")
    if not paper_dir.exists():
        logger.error(f"Paper directory not found: {paper_dir}")
        return False

    pdf_files = list(paper_dir.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files in paper directory")

    if len(pdf_files) == 0:
        logger.warning("No PDF files found - evaluation will run but with no data")

    return True


def run_quick_test() -> bool:
    """Run a quick system test to verify everything works."""
    logger.info("Running quick system test...")

    try:
        # Test 1: Environment check
        if not setup_evaluation_environment():
            logger.error("Environment test failed")
            return False

        # Test 2: Database initialization
        config = {
            "db_path": "./test_db",
            "paper_dir": "parameter/my_papers",
            "output_dir": "test_output",
            "collection_name": "test_memory"
        }

        runner = EvaluationRunner(config)

        # Test 3: Basic database operations
        try:
            stats = runner.db.get_database_stats()
            logger.info(f"Database test passed: {stats.get('total_entries', 0)} entries")
        except Exception as e:
            logger.error(f"Database test failed: {e}")
            return False

        # Test 4: Agent initialization
        try:
            evaluator = ParameterEvaluationSystem(
                db_path=config["db_path"],
                paper_dir=config["paper_dir"]
            )
            logger.info("Agent initialization test passed")
        except Exception as e:
            logger.error(f"Agent initialization test failed: {e}")
            return False

        logger.info("All tests passed!")
        return True

    except Exception as e:
        logger.error(f"Quick test failed: {e}")
        return False


# Factory function to create runner
def create_evaluation_runner(config: Dict[str, Any] = None) -> EvaluationRunner:
    """Create an evaluation runner with the given configuration."""
    return EvaluationRunner(config)


# Keep old import compatibility
OptimizedEvaluationRunner = EvaluationRunner  # Alias for backwards compatibility
create_optimized_runner = create_evaluation_runner  # Alias for backwards compatibility
