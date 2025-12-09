# Parameter Extraction

Vision-driven extraction of quantitative parameters, equations, and text from scientific PDFs. The pipeline renders PDF pages, calls GPT-4o Vision with task-specific prompts, and saves structured outputs for downstream analysis.

## Layout
- `main.py` - CLI entry point that wires PDF input to the vision processor.
- `requirements.txt` - Minimal dependency list for the vision workflow.
- `vision_pdf_extractor_walkthrough.ipynb` - Notebook describing the end-to-end extraction steps.
- `agents/` - LLM agent implementations used for parameter reasoning and orchestration.
- `memory/` - Workspace for intermediate agent memory and artifacts (emptied in git).
- `tumor-tcell/` - Ground-truth documents and notes for benchmarking the extractor.
- `utils/vision_based_processor.py` - `VisionPDFExtractor` helper for rendering pages, calling GPT-4o Vision, and writing JSON/TXT/MD artifacts.
- `papers/` - (user-supplied) PDF inputs. The default script looks for `paper_1.pdf`.
- `output/vision/` - Auto-created folder containing extractor outputs.

## Folder Structure
```
parameter_extraction/
|-- main.py
|-- README.md
|-- requirements.txt
|-- vision_pdf_extractor_walkthrough.ipynb
|-- agents/
|   |-- base_agents.py
|   |-- specialized_agents.py
|-- memory/
|-- output/
|-- papers/
|-- tumor-tcell/
|   |-- tt_bio_mapping_with_queries.md
|   |-- tt_ground_truth.md
|   |-- tumor_tcell_supplemental_information.pdf
|-- utils/
|   |-- vision_based_processor.py
```

## Prerequisites
- Python 3.9+
- Python packages: `pymupdf`, `pillow`, `openai`
- OpenAI API key with access to a vision-capable model (e.g., `gpt-4o`)

Install the dependencies, then expose your API key:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate
pip install pymupdf pillow openai
$env:OPENAI_API_KEY = 'sk-...'    # PowerShell
```

For Bash shells replace the last line with `export OPENAI_API_KEY=sk-...`.

## Running the Vision Processor
1. Place the target PDF inside `papers/` (the default path is `papers/paper_1.pdf`).
2. Choose a processing mode:
   - `parameters` - extract quantitative parameters and their context.
   - `transcribe` - capture verbatim text, tables, and captions.
   - `equations` - focus on LaTeX-ready equation transcription.
3. Call the CLI, which delegates to `VisionPDFExtractor` from `utils/vision_based_processor.py`:

```powershell
python main.py --mode parameters --page 5 --dpi 400
```

Options:
- `--mode` selects one of the prompts above (`parameters` default).
- `--page` is 1-based (use `1` for the first page).
- `--dpi` controls rasterization quality (higher values improve fidelity at the cost of runtime).

The script prints the saved artifact paths. Results land in `output/vision/` as:
- `*.json` - raw metadata + model response.
- `*.txt` - plain text version of the model output (if present).
- `*.md` - Markdown with a short header plus the model response.

## Troubleshooting
- If you see `OPENAI_API_KEY not set`, export the key in your shell and retry.
- `Page index ... out of range` indicates the page number exceeds the PDF length - lower `--page` or use a longer document.
- Rendering errors usually stem from corrupted PDFs; verify the input opens locally.
