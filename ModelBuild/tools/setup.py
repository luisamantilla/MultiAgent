# setup.py - Installation and environment validation script

import os
import sys
import subprocess
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        logger.error("Python 3.8 or higher is required")
        return False
    logger.info(f"Python version: {version.major}.{version.minor}.{version.micro}")
    return True


def install_requirements():
    """Install required packages from requirements.txt."""
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        logger.error("requirements.txt not found")
        return False

    try:
        logger.info("Installing required packages...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ])
        logger.info("Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install requirements: {e}")
        return False


def check_openai_key():
    """Check if OpenAI API key is set."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY environment variable not set")
        logger.info("Set it with: export OPENAI_API_KEY='your-api-key-here'")
        return False

    # Basic validation - should start with 'sk-'
    if not api_key.startswith('sk-'):
        logger.warning("OPENAI_API_KEY format looks incorrect (should start with 'sk-')")
        return False

    logger.info("OPENAI_API_KEY is configured")
    return True


def setup_directories():
    """Create necessary directories."""
    directories = [
        "multi_agent_db",
        "evaluation_results",
        "logs",
        "parameter/my_papers",
        "parameter/query_outputs"
    ]

    for directory in directories:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")


def check_paper_directory():
    """Check if paper directory exists and has PDFs."""
    paper_dir = Path("parameter/my_papers")
    if not paper_dir.exists():
        logger.warning(f"Paper directory not found: {paper_dir}")
        return False

    pdf_files = list(paper_dir.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files in paper directory")

    if len(pdf_files) == 0:
        logger.warning("No PDF files found - add some papers to test the system")
        return False

    return True


def test_imports():
    """Test if all required modules can be imported."""
    required_modules = [
        "openai",
        "chromadb",
        "numpy",
        "scipy",
        "PyPDF2",
        "fitz",  # PyMuPDF
        "pandas"
    ]

    failed_imports = []
    for module in required_modules:
        try:
            __import__(module)
            logger.info(f"✓ {module}")
        except ImportError:
            logger.error(f"✗ {module}")
            failed_imports.append(module)

    if failed_imports:
        logger.error(f"Failed to import: {', '.join(failed_imports)}")
        return False

    logger.info("All required modules imported successfully")
    return True


def test_database_creation():
    """Test database creation and basic operations."""
    try:
        from memory.pdf_vector_db import MultiAgentSharedDatabase

        # Create test database
        test_db = MultiAgentSharedDatabase(
            db_dir="./test_db",
            collection_name="test_collection"
        )

        # Test basic functionality
        stats = test_db.get_database_stats()
        logger.info(f"Database test passed: {stats.get('total_entries', 0)} entries")

        # Cleanup test database
        import shutil
        shutil.rmtree("./test_db", ignore_errors=True)

        return True

    except Exception as e:
        logger.error(f"Database test failed: {e}")
        return False


def run_system_validation():
    """Run comprehensive system validation."""
    logger.info("Starting system validation...")

    checks = [
        ("Python version", check_python_version),
        ("Install requirements", install_requirements),
        ("Test imports", test_imports),
        ("Setup directories", lambda: (setup_directories(), True)[1]),
        ("Check OpenAI key", check_openai_key),
        ("Check paper directory", check_paper_directory),
        ("Test database", test_database_creation)
    ]

    passed = 0
    total = len(checks)

    for check_name, check_func in checks:
        logger.info(f"Running: {check_name}")
        try:
            if check_func():
                logger.info(f"✓ {check_name} passed")
                passed += 1
            else:
                logger.warning(f"✗ {check_name} failed")
        except Exception as e:
            logger.error(f"✗ {check_name} error: {e}")

    logger.info(f"Validation complete: {passed}/{total} checks passed")

    if passed == total:
        logger.info("System is ready for evaluation!")
        print_usage_guide()
        return True
    else:
        logger.warning("Some checks failed. Please address the issues above.")
        return False


def print_usage_guide():
    """Print usage guide for the system."""
    print("\n" + "=" * 80)
    print("MULTI-AGENT PARAMETER EXTRACTION SYSTEM - READY!")
    print("=" * 80)
    print()
    print("Quick Start Guide:")
    print()
    print("1. Test the system:")
    print("   python main.py --test")
    print()
    print("2. Run a single parameter extraction:")
    print("   python main.py --mode single --prompt-id t-cell_01")
    print()
    print("3. Evaluate one cell type:")
    print("   python main.py --mode cell-type --cell-type t-cell")
    print()
    print("4. Run full evaluation (all parameters):")
    print("   python main.py --mode full")
    print()
    print("5. List all available prompts:")
    print("   python main.py --list-prompts")
    print()
    print("Expected Performance:")
    print("- Processing time: 30-60 seconds per parameter")
    print("- Success rate: 80-90% for Level 1 prompts")
    print("- Full evaluation: 2-3 hours for all parameters")
    print()
    print("Troubleshooting:")
    print("- Check logs in: evaluation.log")
    print("- Results saved in: evaluation_results/")
    print("- Database stored in: multi_agent_db/")
    print("=" * 80)


def main():
    """Main setup function."""
    print("Multi-Agent Parameter Extraction System Setup")
    print("=" * 50)

    if run_system_validation():
        print("\nSetup completed successfully!")
        return 0
    else:
        print("\nSetup failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())