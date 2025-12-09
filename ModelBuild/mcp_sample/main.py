# main.py
"""
Fixed main integration module for the Biology Agents Multi-Agent System.
Resolves critical bugs and provides a working unified interface.
Now includes a stateful interactive chat mode (--interactive).
"""

import asyncio
import logging
import sys
import os
import time
import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional, List

# Fix Python path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Configure logging with proper formatting
log_dir = Path("./output/logs")
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_dir / "system.log")
    ]
)
logger = logging.getLogger(__name__)


class BiologyAgentsSystem:
    """
    Fixed main system class with proper error handling and component integration.
    """

    def __init__(self,
                 papers_dir: str = "./papers",
                 output_dir: str = "./output",
                 config: Optional[Dict[str, Any]] = None):
        self.papers_dir = Path(papers_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.config = config or {}

        # Core components (lazy initialization)
        self.agents = None
        self.orchestrator = None
        self.vector_db = None
        self.rag_pipeline = None
        self.mcp_server = None

        # System state
        self.initialized = False
        self.initialization_errors = []
        self.initialization_time = None

        # Create required directories
        self._setup_directories()

    def _setup_directories(self):
        """Setup required directory structure."""
        required_dirs = [
            self.papers_dir,
            self.output_dir,
            self.output_dir / "logs",
            self.output_dir / "generated_repos",
            Path("./cache")
        ]

        for directory in required_dirs:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Directory ready: {directory}")
            except Exception as e:
                logger.warning(f"Could not create directory {directory}: {e}")

    def _check_environment(self) -> bool:
        """Check environment variables and dependencies."""
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            self.initialization_errors.append("OPENAI_API_KEY not found - agents will return stub responses")
            logger.warning("OPENAI_API_KEY not set")
        else:
            logger.info("✓ OpenAI API key found")

        email = os.getenv("EMAIL")
        if email:
            logger.info("✓ Email found for PubMed access")
        else:
            logger.info("Email not set - PubMed search will be limited")

        return True  # Continue even without all env vars

    async def initialize(self,
                         initialize_vector_db: bool = True,
                         initialize_rag: bool = True,
                         initialize_mcp: bool = False) -> bool:
        """Initialize all system components with proper error handling."""
        if self.initialized:
            logger.info("System already initialized")
            return True

        start_time = time.time()
        logger.info("Initializing Biology Agents System...")

        try:
            # Check environment
            self._check_environment()

            # 1. Initialize core agents
            logger.info("Initializing agents...")
            try:
                from agents.specialized_agents import create_all_agents
                self.agents = create_all_agents()
                logger.info(f"✓ Initialized {len(self.agents)} agents: {list(self.agents.keys())}")
            except Exception as e:
                logger.error(f"Agent initialization failed: {e}")
                self.initialization_errors.append(f"Agent initialization failed: {e}")
                return False

            # 2. Initialize vector database
            if initialize_vector_db:
                logger.info("Initializing vector database...")
                try:
                    from memory.pdf_vector_db import PDFVectorDB
                    # Keep constructor signature flexible
                    try:
                        self.vector_db = PDFVectorDB(papers_dir=str(self.papers_dir))
                    except TypeError:
                        # Fallback to original signature if present
                        self.vector_db = PDFVectorDB()

                    # Process existing PDFs
                    if self.papers_dir.exists():
                        pdf_files = list(self.papers_dir.glob("*.pdf"))
                        if pdf_files and hasattr(self.vector_db, "add_pdf_directory"):
                            logger.info(f"Processing {len(pdf_files)} PDF files...")
                            self.vector_db.add_pdf_directory(str(self.papers_dir))
                            if hasattr(self.vector_db, "get_database_stats"):
                                stats = self.vector_db.get_database_stats()
                                logger.info(f"✓ Vector database ready: {stats}")
                        else:
                            logger.info("No PDF files found in papers directory")

                    # Attach to agents
                    for agent in self.agents.values():
                        if hasattr(agent, 'attach_runtime'):
                            agent.attach_runtime(db=self.vector_db)
                    logger.info("✓ Attached vector database to agents")

                except Exception as e:
                    logger.warning(f"Vector database initialization failed: {e}")
                    self.vector_db = None
                    self.initialization_errors.append(f"Vector DB failed: {e}")

            # 3. Initialize RAG pipeline
            if initialize_rag:
                logger.info("Initializing RAG pipeline...")
                try:
                    from tools.tools import RAGPipeline
                    email = os.getenv("EMAIL") or self.config.get("email")
                    llamaparse_key = os.getenv("LLAMAPARSE_API_KEY") or self.config.get("llamaparse_api_key")

                    self.rag_pipeline = RAGPipeline(email=email, llamaparse_api_key=llamaparse_key)
                    logger.info("✓ RAG pipeline initialized")
                except Exception as e:
                    logger.warning(f"RAG pipeline initialization failed: {e}")
                    self.rag_pipeline = None
                    self.initialization_errors.append(f"RAG pipeline failed: {e}")

            # 4. Initialize orchestrator
            logger.info("Initializing orchestrator...")
            try:
                from tools.orchestration import AdaptiveOrchestrator
                self.orchestrator = AdaptiveOrchestrator(self.agents)
                logger.info("✓ Orchestrator initialized")
            except Exception as e:
                logger.error(f"Orchestrator initialization failed: {e}")
                self.initialization_errors.append(f"Orchestrator failed: {e}")
                return False

            # 5. Initialize MCP server if requested
            if initialize_mcp:
                logger.info("Initializing MCP server...")
                try:
                    from agents.mcp_adapters import setup_mcp_orchestrator
                    self.mcp_server = setup_mcp_orchestrator(self.agents)
                    logger.info("✓ MCP server initialized")
                except Exception as e:
                    logger.warning(f"MCP server initialization failed: {e}")
                    self.mcp_server = None
                    self.initialization_errors.append(f"MCP server failed: {e}")

            # Mark as initialized
            self.initialized = True
            self.initialization_time = time.time() - start_time

            logger.info(f"✓ System initialization complete in {self.initialization_time:.2f}s")
            await self.print_system_status()

            return True

        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            self.initialization_errors.append(f"System initialization failed: {e}")
            return False

    async def print_system_status(self):
        """Print comprehensive system status."""
        print("\n" + "=" * 70)
        print("BIOLOGY AGENTS SYSTEM STATUS")
        print("=" * 70)

        # Core components status
        print("CORE COMPONENTS:")
        components = [
            ("Agents", f"{len(self.agents)} ready" if self.agents else "Not available"),
            ("Vector Database", "Ready" if self.vector_db else "Not available"),
            ("RAG Pipeline", "Ready" if self.rag_pipeline else "Not available"),
            ("Orchestrator", "Ready" if self.orchestrator else "Not available"),
            ("MCP Server", "Ready" if self.mcp_server else "Not available")
        ]

        for name, status in components:
            icon = "✓" if "Ready" in status or "ready" in status else "✗"
            print(f"  {icon} {name}: {status}")

        # Database statistics
        if self.vector_db and hasattr(self.vector_db, "get_database_stats"):
            try:
                stats = self.vector_db.get_database_stats()
                print(f"\nDATABASE STATISTICS:")
                print(f"  • Papers: {stats.get('pdf_papers', 0)}")
                print(f"  • Text chunks: {stats.get('total_chunks', 0)}")
                print(f"  • Total entries: {stats.get('total_entries', 0)}")
            except Exception as e:
                print(f"\nDATABASE STATISTICS: Error retrieving stats - {e}")

        # Available agents
        if self.agents:
            print(f"\nAVAILABLE AGENTS:")
            for agent_id, agent in self.agents.items():
                title = getattr(agent, 'title', agent_id)
                try:
                    caps = agent.get_capabilities() if hasattr(agent, 'get_capabilities') else []
                    caps_str = ', '.join(caps[:3]) + ('...' if len(caps) > 3 else '')
                except:
                    caps_str = "capabilities unavailable"
                print(f"  • {agent_id}: {title}")
                print(f"    Capabilities: {caps_str}")

        # Workflow templates
        if self.orchestrator:
            try:
                templates = self.orchestrator.get_available_templates()
                print(f"\nWORKFLOW TEMPLATES:")
                for template in templates:
                    print(f"  • {template}")
            except Exception as e:
                print(f"\nWORKFLOW TEMPLATES: Error retrieving templates - {e}")

        # Directory status
        print(f"\nDIRECTORIES:")
        dirs = [
            ("Papers", self.papers_dir),
            ("Output", self.output_dir),
            ("Logs", self.output_dir / "logs"),
            ("Generated Repos", self.output_dir / "generated_repos")
        ]

        for name, path in dirs:
            exists = path.exists()
            icon = "✓" if exists else "✗"
            count = ""
            if exists and name == "Papers":
                pdf_count = len(list(path.glob("*.pdf"))) if path.exists() else 0
                count = f" ({pdf_count} PDFs)"
            print(f"  {icon} {name}: {path}{count}")

        # Initialization errors/warnings
        if self.initialization_errors:
            print(f"\nINITIALIZATION ISSUES:")
            for error in self.initialization_errors:
                print(f"  ⚠ {error}")

        print(f"\nSYSTEM READY: {self.initialized}")
        print("=" * 70 + "\n")

    async def run_biology_workflow(self,
                                   user_request: str,
                                   workflow_type: str = "auto",
                                   use_rag: bool = True,
                                   output_project_name: Optional[str] = None) -> Dict[str, Any]:
        """Run complete biology modeling workflow with error handling."""
        if not self.initialized:
            await self.initialize()

        if not self.orchestrator:
            raise RuntimeError("Orchestrator not available - system initialization failed")

        start_time = time.time()
        workflow_id = f"workflow_{int(time.time() * 1000)}"

        logger.info(f"Starting workflow {workflow_id}: {user_request[:100]}...")

        try:
            # Prepare context
            context = {"user_request": user_request}

            # Add RAG context if available
            if use_rag and self.rag_pipeline and self.vector_db:
                logger.info("Gathering research context...")
                try:
                    rag_results = await self.rag_pipeline.run_rag_pipeline(
                        user_request, self.vector_db, search_online=True
                    )
                    context["rag_results"] = rag_results
                    logger.info(f"Found {len(rag_results.get('chunks_retrieved', []))} relevant chunks")
                except Exception as e:
                    logger.warning(f"RAG processing failed: {e}")

            # Run workflow
            template_name = None if workflow_type == "auto" else workflow_type

            result = await self.orchestrator.run_adaptive_workflow(
                user_request=user_request,
                template_name=template_name,
                rag_context=context
            )

            # Process results
            execution_time = time.time() - start_time

            if result.get("status") == "success":
                logger.info(f"Workflow {workflow_id} completed successfully in {execution_time:.2f}s")

                # Generate project name
                if not output_project_name:
                    import re
                    words = re.findall(r'\b[a-zA-Z]+\b', user_request.lower())
                    bio_words = [w for w in words if len(w) > 3][:3]
                    output_project_name = "_".join(bio_words) if bio_words else "biology_model"

                # Create summary
                result["execution_summary"] = {
                    "workflow_id": workflow_id,
                    "user_request": user_request,
                    "template_used": result.get("template_used", "auto-selected"),
                    "execution_time": execution_time,
                    "project_name": output_project_name,
                    "rag_enhanced": use_rag and bool(context.get("rag_results")),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }

                # Save execution log
                log_file = self.output_dir / "logs" / f"workflow_{workflow_id}.json"
                try:
                    with open(log_file, 'w') as f:
                        json.dump(result["execution_summary"], f, indent=2, default=str)
                    result["log_file"] = str(log_file)
                    logger.info(f"Execution log saved: {log_file}")
                except Exception as e:
                    logger.warning(f"Could not save execution log: {e}")

            else:
                logger.error(f"Workflow {workflow_id} failed: {result.get('error', 'Unknown error')}")

            result["workflow_id"] = workflow_id
            return result

        except Exception as e:
            error_msg = f"Workflow execution failed: {str(e)}"
            logger.error(error_msg)

            return {
                "workflow_id": workflow_id,
                "status": "failed",
                "error": error_msg,
                "execution_time": time.time() - start_time
            }

    async def quick_agent_test(self, agent_id: str, prompt: str) -> Dict[str, Any]:
        """Quick test of a specific agent."""
        if not self.initialized:
            await self.initialize()

        if agent_id not in self.agents:
            return {"error": f"Agent {agent_id} not found. Available: {list(self.agents.keys())}"}

        agent = self.agents[agent_id]

        try:
            start_time = time.time()

            # Build conversation
            conversation = [
                agent.system_message(),
                {"role": "user", "content": prompt}
            ]

            # Run agent
            result = agent.run(conversation)
            execution_time = time.time() - start_time

            return {
                "agent_id": agent_id,
                "agent_title": getattr(agent, 'title', 'Unknown'),
                "prompt": prompt,
                "result": result,
                "execution_time": execution_time,
                "status": "success"
            }

        except Exception as e:
            return {
                "agent_id": agent_id,
                "error": str(e),
                "status": "failed"
            }

    async def search_papers(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Search papers in vector database."""
        if not self.vector_db:
            return {"error": "Vector database not available"}

        try:
            results = self.vector_db.search_content(query, top_k=max_results)

            return {
                "query": query,
                "results_count": len(results),
                "results": results,
                "database_stats": self.vector_db.get_database_stats()
            }

        except Exception as e:
            return {"error": f"Search failed: {str(e)}"}

    # ---------- NEW: Stateful interactive chat ----------
    async def interactive_chat(self, use_rag: bool = True):
        """
        Start a stateful interactive session.
        Maintains context across turns (previous results + optional RAG).
        Commands:
          :status  - show system status
          :save    - save last summary to output/logs/chat_<timestamp>.json
          :quit    - exit session
        """
        if not self.initialized:
            await self.initialize()

        print("\nInteractive Biology Agents Chat")
        print("Type your prompt. Commands: :status  :save  :quit\n")

        session_context: Dict[str, Any] = {"previous_results": {}}
        last_summary: Optional[Dict[str, Any]] = None

        while True:
            try:
                user_in = input("> ").strip()
                if not user_in:
                    continue

                if user_in.lower() in (":quit", ":exit"):
                    print("Bye!")
                    break
                if user_in.lower() == ":status":
                    await self.print_system_status()
                    continue
                if user_in.lower() == ":save":
                    if last_summary:
                        ts = time.strftime("%Y%m%d_%H%M%S")
                        p = self.output_dir / "logs" / f"chat_{ts}.json"
                        with open(p, "w") as f:
                            json.dump(last_summary, f, indent=2)
                        print(f"Saved: {p}")
                    else:
                        print("Nothing to save yet.")
                    continue

                # Optionally enrich each turn with RAG
                rag_ctx = {}
                if use_rag and self.rag_pipeline and self.vector_db:
                    try:
                        rag_results = await self.rag_pipeline.run_rag_pipeline(
                            user_in, self.vector_db, search_online=True
                        )
                        rag_ctx["rag_results"] = rag_results
                    except Exception as e:
                        logger.warning(f"RAG failed this turn: {e}")

                # Merge session context
                merged_ctx = {**session_context, **rag_ctx, "user_request": user_in}

                # Run one full workflow per turn (auto template)
                result = await self.orchestrator.run_adaptive_workflow(
                    user_request=user_in,
                    rag_context=merged_ctx
                )

                # Update session context with latest results
                if isinstance(result, dict) and "results" in result:
                    session_context["previous_results"] = {
                        **session_context.get("previous_results", {}),
                        **result["results"]
                    }

                # Display concise step outputs
                if result.get("status") == "success":
                    print("\n--- Steps ---")
                    for step_id, step_res in result.get("results", {}).items():
                        content = step_res.get("content", "")
                        preview = content.strip().splitlines()[0][:180] if isinstance(content, str) else str(content)[:180]
                        print(f"[{step_id}] {preview}{'…' if len(preview) == 180 else ''}")
                    print("-------------\n")
                else:
                    print(f"[ERROR] {result.get('error', 'Unknown error')}")

                # Keep a simple summary for :save
                last_summary = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "user_request": user_in,
                    "status": result.get("status"),
                    "template_used": result.get("template_used"),
                    "results_keys": list(result.get("results", {}).keys()) if isinstance(result.get("results"), dict) else [],
                }

            except KeyboardInterrupt:
                print("\nInterrupted. Type :quit to exit.")
            except Exception as e:
                print(f"[ERROR] {e}")

# Command Line Interface
async def main():
    """Main entry point with command line interface."""
    parser = argparse.ArgumentParser(description="Biology Agents Multi-Agent System")
    parser.add_argument("--papers-dir", default="./papers", help="Papers directory")
    parser.add_argument("--output-dir", default="./output", help="Output directory")
    parser.add_argument("--no-vector-db", action="store_true", help="Skip vector database")
    parser.add_argument("--no-rag", action="store_true", help="Skip RAG pipeline")
    parser.add_argument("--enable-mcp", action="store_true", help="Enable MCP server")
    parser.add_argument("--test-agent", help="Test specific agent with prompt")
    parser.add_argument("--search", help="Search papers in database")
    parser.add_argument("--workflow", help="Run biology modeling workflow")
    parser.add_argument("--workflow-type", default="auto", choices=["auto", "standard", "rapid", "literature"])
    parser.add_argument("--interactive", action="store_true", help="Start interactive chat session")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize system
    system = BiologyAgentsSystem(
        papers_dir=args.papers_dir,
        output_dir=args.output_dir
    )

    try:
        success = await system.initialize(
            initialize_vector_db=not args.no_vector_db,
            initialize_rag=not args.no_rag,
            initialize_mcp=args.enable_mcp
        )

        if not success:
            print("System initialization failed!")
            sys.exit(1)

        # New interactive mode
        if args.interactive:
            await system.interactive_chat(use_rag=not args.no_rag)
            return

        # Execute requested operation
        if args.test_agent:
            if ":" not in args.test_agent:
                print("Usage: --test-agent agent_id:prompt")
                sys.exit(1)

            agent_id, prompt = args.test_agent.split(":", 1)
            result = await system.quick_agent_test(agent_id, prompt)
            print(json.dumps(result, indent=2, default=str))

        elif args.search:
            result = await system.search_papers(args.search)
            print(json.dumps(result, indent=2, default=str))

        elif args.workflow:
            result = await system.run_biology_workflow(
                user_request=args.workflow,
                workflow_type=args.workflow_type,
                use_rag=not args.no_rag
            )
            print(json.dumps(result, indent=2, default=str))

        else:
            # Legacy interactive commands (kept for backward compatibility)
            print("\nBiology Agents System Ready!")
            print("Available commands:")
            print("  workflow: <request>  - Run biology modeling workflow")
            print("  test: <agent_id> <prompt>  - Test specific agent")
            print("  search: <query>  - Search papers database")
            print("  status  - Show system status")
            print("  quit  - Exit system")
            print("Tip: use --interactive for the new chat mode.\n")

            while True:
                try:
                    command = input("\n> ").strip()
                    if not command:
                        continue

                    if command.lower() in ['quit', 'exit', 'q']:
                        break
                    elif command.lower() == 'status':
                        await system.print_system_status()
                    elif command.startswith('workflow:'):
                        request = command[9:].strip()
                        result = await system.run_biology_workflow(request)
                        print(json.dumps(result, indent=2, default=str))
                    elif command.startswith('test:'):
                        parts = command[5:].strip().split(' ', 1)
                        if len(parts) != 2:
                            print("Usage: test: <agent_id> <prompt>")
                            continue
                        result = await system.quick_agent_test(parts[0], parts[1])
                        print(json.dumps(result, indent=2, default=str))
                    elif command.startswith('search:'):
                        query = command[7:].strip()
                        result = await system.search_papers(query)
                        print(json.dumps(result, indent=2, default=str))
                    else:
                        print("Unknown command. Type 'quit' to exit.")

                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"Error: {e}")

    except KeyboardInterrupt:
        logger.info("System interrupted by user")
    except Exception as e:
        logger.error(f"System error: {e}")
        sys.exit(1)
    finally:
        print("\nShutting down Biology Agents System...")


if __name__ == "__main__":
    asyncio.run(main())
