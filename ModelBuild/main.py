# main.py

import os
import sys
import asyncio
import argparse
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from tools.runner import EvaluationRunner, setup_evaluation_environment, run_quick_test
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all required packages are installed and the file structure is correct.")
    sys.exit(1)

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


def print_usage_examples():
    """Print usage examples for the evaluation system."""
    print("\n" + "=" * 80)
    print("MULTI-AGENT PARAMETER EXTRACTION EVALUATION SYSTEM")
    print("=" * 80)
    print("Usage Examples:")
    print()
    print("1. Full evaluation (all parameters):")
    print("   python main.py --mode full")
    print()
    print("2. Evaluate T-cell parameters only:")
    print("   python main.py --mode cell-type --cell-type t-cell")
    print()
    print("3. Evaluate single parameter:")
    print("   python main.py --mode single --prompt-id t-cell_01")
    print()
    print("4. List all available prompts:")
    print("   python main.py --list-prompts")
    print()
    print("5. Setup database only:")
    print("   python main.py --mode setup-only")
    print()
    print("6. Run quick system test:")
    print("   python main.py --test")
    print("=" * 80)


def validate_arguments(args):
    """Validate command line arguments."""
    errors = []

    if args.mode == "single" and not args.prompt_id:
        errors.append("--prompt-id is required when using --mode single")

    if args.mode == "cell-type" and not args.cell_type:
        errors.append("--cell-type is required when using --mode cell-type")

    if args.cell_type and args.mode != "cell-type":
        errors.append("--cell-type can only be used with --mode cell-type")

    if args.prompt_id and args.mode != "single":
        errors.append("--prompt-id can only be used with --mode single")

    return errors


async def main():
    """Main function to orchestrate the evaluation system."""
    parser = argparse.ArgumentParser(
        description="Multi-Agent Parameter Extraction Evaluation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode full                             # Run complete evaluation
  python main.py --mode cell-type --cell-type t-cell    # Evaluate T-cell parameters only
  python main.py --mode single --prompt-id t-cell_01    # Evaluate single parameter
  python main.py --list-prompts                         # Show all available prompts
  python main.py --test                                 # Run quick system test
        """
    )

    parser.add_argument("--mode", choices=["full", "single", "cell-type", "setup-only"],
                        default=None, help="Evaluation mode")
    parser.add_argument("--cell-type", choices=["t-cell", "tumor", "dendritic", "lymph_node"],
                        help="Specific cell type to evaluate")
    parser.add_argument("--prompt-id", help="Specific prompt ID to evaluate")
    parser.add_argument("--db-path", default="./multi_agent_db", help="Database path")
    parser.add_argument("--paper-dir", default="parameter/my_papers", help="Papers directory")
    parser.add_argument("--output-dir", default="evaluation_results", help="Output directory")
    parser.add_argument("--list-prompts", action="store_true", help="List all available prompts")
    parser.add_argument("--test", action="store_true", help="Run quick system test")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    parser.add_argument("--force-rebuild", action="store_true", help="Force rebuild database")

    args = parser.parse_args()

    # Handle no arguments - show usage
    if len(sys.argv) == 1:
        print_usage_examples()
        return

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Handle quick test
    if args.test:
        logger.info("Running quick system test...")
        try:
            if run_quick_test():
                logger.info("System test passed!")
            else:
                logger.error("System test failed!")
                sys.exit(1)
        except Exception as e:
            logger.error(f"System test error: {e}")
            sys.exit(1)
        return

    # Validate arguments
    if not args.list_prompts and not args.mode:
        logger.error("Must specify either --mode or --list-prompts")
        print_usage_examples()
        sys.exit(1)

    validation_errors = validate_arguments(args)
    if validation_errors:
        for error in validation_errors:
            logger.error(error)
        sys.exit(1)

    # Check environment setup
    try:
        if not setup_evaluation_environment():
            logger.error("Environment setup failed. Please check the requirements.")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Environment setup error: {e}")
        sys.exit(1)

    # Create configuration
    config = {
        "db_path": args.db_path,
        "paper_dir": args.paper_dir,
        "output_dir": args.output_dir,
        "collection_name": "evaluation_memory",
        "force_rebuild_db": args.force_rebuild
    }

    # Initialize runner
    try:
        runner = EvaluationRunner(config)
    except Exception as e:
        logger.error(f"Failed to initialize runner: {e}")
        sys.exit(1)

    # Handle list prompts
    if args.list_prompts:
        try:
            runner.print_available_prompts()
        except Exception as e:
            logger.error(f"Failed to list prompts: {e}")
            sys.exit(1)
        return

    # Run evaluation based on mode
    success = False
    try:
        if args.mode == "full":
            logger.info("Starting full system evaluation...")
            success = await runner.run_full_evaluation()

        elif args.mode == "cell-type":
            logger.info(f"Starting {args.cell_type} evaluation...")
            await runner.run_cell_type_evaluation(args.cell_type)
            success = True

        elif args.mode == "single":
            logger.info(f"Starting single parameter evaluation...")
            success = await runner.run_single_parameter_evaluation(args.prompt_id)

        elif args.mode == "setup-only":
            logger.info("Setting up database only...")
            success = await runner.initialize_database()

        if success:
            logger.info("Evaluation completed successfully!")
            logger.info(f"Check results in: {config['output_dir']}")
        else:
            logger.error("Evaluation failed!")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("\nEvaluation interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error during evaluation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


def run_test_mode():
    """Run system in test mode with minimal operations."""
    logger.info("Running in test mode...")

    try:
        # Test environment setup
        if not setup_evaluation_environment():
            logger.error("Environment setup failed")
            return False

        # Run quick test
        if run_quick_test():
            logger.info("Test mode completed successfully!")
            return True
        else:
            logger.error("Test mode failed!")
            return False
    except Exception as e:
        logger.error(f"Test mode error: {e}")
        return False


if __name__ == "__main__":
    try:
        # Check for legacy test mode
        if len(sys.argv) > 1 and sys.argv[1] == "--test":
            if run_test_mode():
                sys.exit(0)
            else:
                sys.exit(1)
        else:
            asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nProgram interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback

        logger.error(traceback.format_exc())
        sys.exit(1)
