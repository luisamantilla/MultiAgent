import asyncio
from paperqa import Docs, Settings
from pathlib import Path
import sys
from datetime import datetime

async def run_local_query(papers_folder: str, output_dir: str, query: str):
    docs = Docs()

    # Load all PDF files from the local folder
    paper_dir = Path(papers_folder)
    pdf_files = sorted(paper_dir.glob("*.pdf"))
    if not pdf_files:
        print("No PDF files found in", papers_folder)
        return

    for pdf in pdf_files:
        print(f"Adding {pdf.name} to the knowledge base...")
        await docs.aadd(str(pdf))

    # Configure settings
    settings = Settings(
        temperature=0.3,
        llm_config={"rate_limit": {"gpt-4o-2024-11-20": "200000 per 1 minute"}}
    )

    print("\nSending query to local PaperQA DB...")
    response = await docs.aquery(query, settings=settings)
    summary = response.formatted_answer.strip()

    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(output_dir) / f"query_{timestamp}.md"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Save output
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# PaperQA Query Results (Local Papers)\n\n")
        f.write(f"**Query:**\n```\n{query.strip()}\n```\n\n")
        f.write(summary)

    print(f"\nuery completed. Output saved to: {output_file}")

if __name__ == "__main__":
    print("Enter your query (Press Enter when done):")
    user_query = sys.stdin.read().strip()

    asyncio.run(
        run_local_query(
            papers_folder = "../my_papers",
            output_dir="../query_outputs",
            query=user_query
        )
    )
