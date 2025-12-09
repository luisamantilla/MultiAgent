#!/usr/bin/env python3
"""
Integrated MCP Setup Script for Biology Agents System
Fixes critical setup and configuration issues.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
import subprocess
import importlib.util

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MCPSystemSetup:
    """Setup and configuration for the Biology Agents MCP system."""

    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir).resolve()
        self.errors = []
        self.warnings = []

    def check_dependencies(self):
        """Check for required dependencies."""
        required_packages = [
            'openai', 'chromadb', 'PyPDF2', 'pymupdf', 'fastapi',
            'uvicorn', 'aiohttp', 'pandas', 'numpy', 'asyncio'
        ]

        missing = []
        for package in required_packages:
            try:
                importlib.import_module(package.replace('-', '_'))
                logger.info(f"✓ {package} found")
            except ImportError:
                missing.append(package)
                logger.warning(f"✗ {package} missing")

        if missing:
            self.errors.append(f"Missing packages: {', '.join(missing)}")
            logger.error(f"Install missing packages: pip install {' '.join(missing)}")

    def check_directory_structure(self):
        """Verify directory structure is correct."""
        required_dirs = [
            'agents', 'mcp', 'tools', 'memory', 'papers', 'output',
            'output/logs', 'output/generated_repos'
        ]

        for dir_path in required_dirs:
            full_path = self.base_dir / dir_path
            if not full_path.exists():
                full_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {full_path}")
            else:
                logger.info(f"✓ Directory exists: {dir_path}")

    def check_environment_variables(self):
        """Check for required environment variables."""
        env_vars = {
            'OPENAI_API_KEY': 'Required for agent LLM calls',
            'EMAIL': 'Optional: for PubMed API access',
            'LLAMAPARSE_API_KEY': 'Optional: for advanced PDF parsing'
        }

        for var, description in env_vars.items():
            if os.getenv(var):
                logger.info(f"✓ {var} found")
            else:
                if var == 'OPENAI_API_KEY':
                    self.errors.append(f"Missing required environment variable: {var}")
                else:
                    self.warnings.append(f"Missing optional environment variable: {var} - {description}")

    def fix_import_paths(self):
        """Fix import path issues in Python files."""
        # Add current directory to Python path
        init_files = [
            'agents/__init__.py',
            'mcp/__init__.py',
            'tools/__init__.py',
            'memory/__init__.py'
        ]

        for init_file in init_files:
            init_path = self.base_dir / init_file
            if not init_path.exists():
                init_path.parent.mkdir(parents=True, exist_ok=True)
                init_path.write_text('# Package initialization\n')
                logger.info(f"Created {init_file}")

    def create_config_file(self):
        """Create default configuration file."""
        config_content = '''# Biology Agents MCP System Configuration

# System Settings
PAPERS_DIR = "./papers"
OUTPUT_DIR = "./output" 
CACHE_DIR = "./cache"

# MCP Server Settings
MCP_HOST = "127.0.0.1"
MCP_PORT = 8000

# Agent Settings  
DEFAULT_MODEL = "gpt-4o"
AGENT_TIMEOUT = 300
MAX_RETRIES = 3

# Vector Database Settings
DB_COLLECTION_NAME = "papers"
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200

# RAG Settings
MAX_SEARCH_RESULTS = 10
CONTEXT_LENGTH = 4000
'''

        config_path = self.base_dir / "config.py"
        if not config_path.exists():
            config_path.write_text(config_content)
            logger.info("Created config.py")

    async def test_system_components(self):
        """Test that system components can be imported and initialized."""
        test_results = {}

        # Test agent imports
        try:
            from agents.specialized_agents import create_all_agents
            agents = create_all_agents()
            test_results['agents'] = f"✓ Created {len(agents)} agents"
            logger.info(test_results['agents'])
        except Exception as e:
            test_results['agents'] = f"✗ Agent creation failed: {e}"
            self.errors.append(test_results['agents'])

        # Test vector database
        try:
            from memory.pdf_vector_db import PDFVectorDB
            db = PDFVectorDB()
            test_results['vector_db'] = "✓ Vector database initialized"
            logger.info(test_results['vector_db'])
        except Exception as e:
            test_results['vector_db'] = f"✗ Vector DB failed: {e}"
            self.errors.append(test_results['vector_db'])

        # Test orchestrator
        try:
            from tools.orchestration import AdaptiveOrchestrator
            if 'agents' in locals():
                orchestrator = AdaptiveOrchestrator(agents)
                test_results['orchestrator'] = "✓ Orchestrator initialized"
                logger.info(test_results['orchestrator'])
            else:
                test_results['orchestrator'] = "✗ Cannot test orchestrator without agents"
        except Exception as e:
            test_results['orchestrator'] = f"✗ Orchestrator failed: {e}"
            self.errors.append(test_results['orchestrator'])

        return test_results

    async def run_complete_setup(self):
        """Run complete system setup and verification."""
        logger.info("Starting Biology Agents MCP System Setup...")

        # Check dependencies
        self.check_dependencies()

        # Setup directory structure
        self.check_directory_structure()

        # Check environment
        self.check_environment_variables()

        # Fix imports
        self.fix_import_paths()

        # Create config
        self.create_config_file()

        # Test components
        test_results = await self.test_system_components()

        # Print summary
        print("\n" + "=" * 60)
        print("SETUP SUMMARY")
        print("=" * 60)

        if self.errors:
            print("ERRORS (must fix):")
            for error in self.errors:
                print(f"  ✗ {error}")
        else:
            print("✓ No critical errors found")

        if self.warnings:
            print("\nWARNINGS (recommended):")
            for warning in self.warnings:
                print(f"  ⚠ {warning}")

        print(f"\nCOMPONENT TESTS:")
        for component, result in test_results.items():
            print(f"  {result}")

        success = len(self.errors) == 0
        print(f"\nSetup {'SUCCESSFUL' if success else 'FAILED'}")
        print("=" * 60)

        return success


def main():
    """Main setup function."""
    setup = MCPSystemSetup()

    try:
        success = asyncio.run(setup.run_complete_setup())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Setup failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()