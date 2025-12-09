# parameter/paperqa_wrapper.py

from paperqa import Docs, Settings
import asyncio
from pathlib import Path

async def extract_parameter_summary(pdf_path: str, output_path: str):
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
- Parameter Value (numeric, or write “varies” or “N/A”)
- Units (e.g., cells/min, mm²/s, molecules/min)
- Reference or Modeling Context (e.g., source, purpose, calibration info)

Output as a Markdown table. Example format:

| Interaction | Parameter Name | Parameter Value | Units | Reference or Context |
|-------------|----------------|-----------------|-------|-----------------------|
| Virus diffuses through tissue | Diffusion coefficient | 0.0119 | mm²/s | Sego et al. 2022, page 5 |
| CD8+ T cells chemotax | Chemotactic sensitivity | 10,000 | unitless | Reflects stronger migration sensitivity |
"""

    print("Sending query to PaperQA...")
    response = await docs.aquery(query, settings=settings)
    summary = response.formatted_answer.strip()

    # Save to markdown
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# Parameter Summary Extracted from PDF\n\n")
        f.write(f"**Query:**\n```\n{query.strip()}\n```\n\n")
        f.write(summary)

    print(f"Markdown summary saved to: {output_path}")

if __name__ == "__main__":
    asyncio.run(
        extract_parameter_summary(
            "parameter/my_papers/paper_1.pdf",
            "parameter/parameter_summary.md"
        )
    )
