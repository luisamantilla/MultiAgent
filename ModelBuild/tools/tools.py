# tools/tools.py

import asyncio
from pathlib import Path
import os
import sys

# Add the project root to the Python path to handle imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from paperqa import Docs, Settings

    PAPERQA_AVAILABLE = True
except ImportError:
    PAPERQA_AVAILABLE = False
    print("Warning: paperqa not available. Install with: pip install paperqa")


async def extract_parameter_summary(pdf_path: str, output_path: str) -> None:
    """
    Use PaperQA to extract biological parameters from a given PDF.

    Parameters:
        pdf_path (str): Path to the input PDF file.
        output_path (str): Path to save the Markdown summary.

    This function saves a structured table of extracted parameters to output_path.
    """
    if not PAPERQA_AVAILABLE:
        # Fallback implementation when paperqa is not available
        print(f"PaperQA not available. Creating placeholder summary for: {pdf_path}")

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write("# Parameter Summary Extracted from PDF\n\n")
            f.write(f"**Source:** {pdf_path}\n\n")
            f.write("**Note:** PaperQA not available. This is a placeholder summary.\n\n")
            f.write("| Interaction | Parameter Name | Parameter Value | Units | Reference or Context |\n")
            f.write("|-------------|----------------|-----------------|-------|-----------------------|\n")
            f.write("| Placeholder | Placeholder | Placeholder | Placeholder | Placeholder |\n")

        print(f"Saved placeholder parameter summary to: {output_path}")
        return

    # Check if PDF file exists
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    try:
        docs = Docs()
        await docs.aadd(pdf_path)

        settings = Settings(
            temperature=0.3,
            llm_config={"rate_limit": {"gpt-4o-2024-11-20": "30000 per 1 minute"}}
        )

        query = """
Extract all biological parameter values from the paper and organize them in a structured Markdown table.

For each parameter, include the following columns:
- Interaction (short description of the biological interaction)
- Parameter Name (e.g., infection rate, diffusion coefficient)
- Parameter Value (numeric, or write "varies" or "N/A")
- Units (e.g., cells/min, mm²/s, molecules/min)
- Reference or Modeling Context (e.g., source, purpose, calibration info)

Output as a Markdown table. Example format:

| Interaction | Parameter Name | Parameter Value | Units | Reference or Context |
|-------------|----------------|-----------------|-------|-----------------------|
| Virus diffuses through tissue | Diffusion coefficient | 0.0119 | mm²/s | Sego et al. 2022, page 5 |
| CD8+ T cells chemotax | Chemotactic sensitivity | 10,000 | unitless | Reflects stronger migration sensitivity |
"""

        print(f"Sending query to PaperQA for: {pdf_path}")
        response = await docs.aquery(query, settings=settings)
        summary = response.formatted_answer.strip()

        # Save to markdown
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write("# Parameter Summary Extracted from PDF\n\n")
            f.write(f"**Source:** {pdf_path}\n\n")
            f.write(f"**Query:**\n```\n{query.strip()}\n```\n\n")
            f.write("**Extracted Parameters:**\n\n")
            f.write(summary)

        print(f"Saved parameter summary to: {output_path}")

    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {e}")
        # Create error summary file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write("# Parameter Summary Extracted from PDF\n\n")
            f.write(f"**Source:** {pdf_path}\n\n")
            f.write(f"**Error:** {str(e)}\n\n")
            f.write("Parameter extraction failed.\n")

        raise


def extract_parameters_from_pdf(pdf_path: str, output_path: str) -> None:
    """
    Wrapper to run the async PaperQA extraction synchronously.

    Parameters:
        pdf_path (str): PDF file path.
        output_path (str): Markdown output path.
    """
    asyncio.run(extract_parameter_summary(pdf_path, output_path))


def get_pdf_files(directory: str) -> list:
    """
    Get all PDF files from a directory.

    Parameters:
        directory (str): Directory path to search for PDFs

    Returns:
        list: List of PDF file paths
    """
    pdf_directory = Path(directory)
    if not pdf_directory.exists():
        print(f"Directory not found: {directory}")
        return []

    pdf_files = list(pdf_directory.glob("*.pdf"))
    return [str(pdf_file) for pdf_file in pdf_files]


def create_output_directory(output_path: str) -> None:
    """
    Create output directory if it doesn't exist.

    Parameters:
        output_path (str): Output directory path
    """
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)


# Optional CLI execution
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract parameters from PDF using PaperQA")
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument("output_path", help="Path to output markdown file")

    args = parser.parse_args()

    try:
        extract_parameters_from_pdf(args.pdf_path, args.output_path)
        print("Parameter extraction completed successfully!")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)