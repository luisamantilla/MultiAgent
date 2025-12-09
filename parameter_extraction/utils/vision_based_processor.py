#!/usr/bin/env python3
"""
utils/vision_based_processor.py

Vision-based PDF extraction using GPT-4o.
- Renders a PDF page to an image (PyMuPDF)
- Sends the image to GPT-4o with a chosen prompt
- Saves JSON, TXT (raw), and MD (with a small header)

Requirements:
  pip install pymupdf pillow openai

Environment:
  - Set your OpenAI key before running:
      Windows (cmd):   set OPENAI_API_KEY=sk-...
      Windows (PowerShell): $env:OPENAI_API_KEY="sk-..."
      macOS/Linux:     export OPENAI_API_KEY=sk-...
"""

from __future__ import annotations
from pathlib import Path
from io import BytesIO
from typing import Dict, Any, Optional
import base64
import json
import os

import fitz  # PyMuPDF
from PIL import Image
from openai import OpenAI


# ---------------------- Prompt presets ----------------------

TRANSCRIBE_PROMPT = """\
You will receive a single page image from a scientific PDF.

TASK: Transcribe ALL VISIBLE TEXT from the image as faithfully as possible.
- Preserve reading order and section boundaries.
- Include headers, footers, captions, axis labels, footnotes, and references.
- Keep original casing, punctuation, and symbols.
- If there are tables, reproduce them in Markdown tables (or TSV if necessary).
- If there are equations, reproduce them in LaTeX (inline or display) as they appear.
- Keep lists as lists. Join hyphenated line-breaks only when clearly a soft wrap.
- Do NOT summarize or add commentary—output only the transcribed text.

Output: plain text (with Markdown/LaTeX where applicable). No preface/explanation.
"""

PARAMETERS_PROMPT = PARAMETERS_PROMPT = """\
You will receive a single page image from a scientific PDF.

TASK: Extract ALL quantitative parameters, values, and data visible on this page.

Return a valid JSON object with the following top-level keys:
- "tables": parameters extracted from tables
- "figures": parameters extracted from figures or graphs
- "text": parameters extracted from prose
- "equations": parameters extracted from equations
- "summary": a 3–4 sentence plain text summary of the key findings

Each parameter inside tables/figures/text/equations must be a JSON object with:
- "parameter": name or description
- "value": numerical value(s) or expression
- "units": units if visible, otherwise null
- "context": what it represents

Example format:
{
  "tables": [
    {"parameter": "PD-L1", "value": "2.5", "units": "ng/mL", "context": "immune checkpoint"}
  ],
  "figures": [
    {"parameter": "Survival rate", "value": "60%", "units": "percent", "context": "Kaplan-Meier curve"}
  ],
  "text": [
    {"parameter": "R0", "value": "3.2", "units": null, "context": "basic reproduction number"}
  ],
  "equations": [
    {"parameter": "dN/dt", "value": "N*r*(1-N/K)", "units": "model equation", "context": "logistic growth"}
  ],
  "summary": "This page reports immune checkpoint values, survival data from figures, and constants such as R0."
}

IMPORTANT:
- Respond **only** with a valid JSON object. 
- Do not include triple backticks, the word "json", or any explanation.
- Output must start with '{' and end with '}'.

"""


EQUATIONS_PROMPT = """
You will receive a scientific PDF page image with equations/tables.
Domain constraints:
- Epithelial states: \hat{H} (uninfected), \hat{I} (infected), \hat{D} (dead)
- Immune cells: M (macrophage), K (NK cell), E (CD8^+ T cell)
- Use hats exactly where present. Do NOT substitute Y for H.
Task:
1) Transcribe all equations and table formula columns in LaTeX.
2) Preserve subscripts/superscripts, Greek letters (β, γ, μ), fractions.
3) Keep variable names exactly as in the image; include hats, bars, dots.
4) If unsure between similar glyphs (H vs Y, v vs \nu, 0 vs O), choose the one
   consistent with the above legend and typical influenza models.
Output: LaTeX only, no commentary.
"""

# ---------------------- Extractor class ----------------------

class VisionPDFExtractor:
    """
    Extract content from a PDF page using GPT-4o vision.

    Args:
        model: OpenAI vision-capable model (default "gpt-4o")
        mode:  "parameters" | "transcribe" (controls prompt)
    """

    def __init__(self, model: str = "gpt-4o", mode: str = "parameters"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set. Set it in your environment before running."
            )
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.mode = mode
        if mode == "parameters":
            self.prompt = PARAMETERS_PROMPT
        elif mode == "transcribe":
            self.prompt = TRANSCRIBE_PROMPT
        elif mode == "equations":
            self.prompt = EQUATIONS_PROMPT
        else:
            raise ValueError("mode must be parameters|transcribe|equations")


    # -------- Render: PDF page -> base64 PNG --------
    @staticmethod
    def _page_to_image_b64(pdf_path: Path, page_index: int, dpi: int = 200) -> str:
        """
        Render a PDF page (0-indexed) to a base64-encoded PNG string.
        """
        doc = fitz.open(pdf_path.as_posix())
        try:
            if page_index < 0 or page_index >= len(doc):
                raise IndexError(f"Page index {page_index} out of range for {pdf_path.name}")
            page = doc[page_index]
            mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
            pix = page.get_pixmap(matrix=mat)
            # Convert to PIL (normalizes format) and re-encode to PNG
            img = Image.open(BytesIO(pix.tobytes("png")))
            buf = BytesIO()
            img.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("utf-8")
        finally:
            doc.close()

    # -------- LLM call --------
    '''
    Modified so that this function outputs JSON format, categorizes based off of form it extracted
    parameter from, and separates raw vs. parsed content.
    - raw_content keeps the original string for debugging
    - parsed is a dict read to immediately use
    '''
    def _vision_call(self, image_b64: str) -> Dict[str, Any]:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0.1,
            max_tokens=3000,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": self.prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}", "detail": "high"}}
                ]
            }]
        )
        raw_content = (resp.choices[0].message.content or "").strip()

        try:
            parsed = json.loads(raw_content)
            return {"success": True, "content": parsed, "raw": raw_content, "model": self.model}
        except json.JSONDecodeError as e:
            return {"success": False, "error": f"Invalid JSON: {e}", "raw": raw_content, "model": self.model}
        
        print("\n\n==== RAW MODEL OUTPUT START ====\n")
        print(raw_content[:2000])  # show first 2000 chars
        print("\n==== RAW MODEL OUTPUT END ====\n")



    # -------- One-page pipeline --------
    def process_page(
        self,
        pdf_path: Path,
        page_index: int,
        dpi: int = 400,
        page_note: str = "",
    ) -> Dict[str, Any]:
        """
        Process one page: render -> call vision -> return dict.
        """
        try:
            img_b64 = self._page_to_image_b64(pdf_path, page_index, dpi=dpi)
        except Exception as e:
            return {"success": False, "error": f"Render failed: {e}", "page": page_index + 1}

        try:
            result = self._vision_call(img_b64)
            result.update({"page": page_index + 1, "note": page_note, "mode": self.mode})
            return result
        except Exception as e:
            return {"success": False, "error": f"Vision call failed: {e}", "page": page_index + 1}

    # -------- Save outputs --------
    @staticmethod
    def save_results(result: Dict[str, Any], out_dir: Path, stem: str, mode: str) -> Dict[str, Optional[Path]]:
        """
        Save:
          - JSON: full result (metadata + content)
          - TXT:  raw LLM content (if present)
          - MD:   Markdown with a small header (if present)
        Filenames include the mode to avoid clobbering.
        """
        out_dir.mkdir(parents=True, exist_ok=True)

        json_path = out_dir / f"{stem}_{mode}_vision.json"
        txt_path = out_dir / f"{stem}_{mode}_vision.txt"
        md_path = out_dir / f"{stem}_{mode}_vision.md"

        # JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        content = result.get("content") or ""
        if content:
            # Convert dict to a formatted string if needed
            if isinstance(content, dict):
                text_output = json.dumps(content, indent=2, ensure_ascii=False)
            else:
                text_output = str(content)

            # TXT
            txt_path.write_text(text_output, encoding="utf-8")


            # MD (with a short header for traceability)
            header = [
                "# Vision Extraction",
                "",
                f"- **Model:** {result.get('model', 'unknown')}",
                f"- **Page:** {result.get('page', '?')}",
                f"- **Mode:** {result.get('mode', mode)}",
            ]
            note = result.get("note")
            if note:
                header.append(f"- **Note:** {note}")
            header += ["", "---", "", ""]
            md_path.write_text("\n".join(header) + content, encoding="utf-8")

        return {
            "json": json_path,
            "txt": txt_path if content else None,
            "md": md_path if content else None,
        }

#added main() block
# ---------------------- Run standalone ----------------------
if __name__ == "__main__":
    import fitz
    import datetime

    # Path to your PDF file
    pdf_path = Path("parameter_extraction/papers/paper_1.pdf")
    print(">>> Using PDF:", pdf_path)
    print(">>> Exists:", pdf_path.exists())

    # Create the extractor instance
    extractor = VisionPDFExtractor(model="gpt-4o", mode="parameters")

    # Count total pages
    total_pages = len(fitz.open(pdf_path))
    pages_to_process = [p for p in range(total_pages)]  # process all pages

    # Process each page
    for page in pages_to_process:
        print(f"\n=== Processing page {page + 1} ===")
        result = extractor.process_page(pdf_path, page_index=page, page_note=f"Page {page + 1}")

        # Create results folder if it doesn't exist
        os.makedirs("parameter_extraction/results", exist_ok=True)

        # Save JSON results for each page
        if result.get("success"):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            out = f"parameter_extraction/results/{timestamp}_page{page + 1}.json"
            with open(out, "w") as f:
                json.dump(result["content"], f, indent=2)
            print(f"✅ Page {page + 1} JSON saved to {out}")
        else:
            print(f"❌ Page {page + 1} failed:", result.get("error"))
