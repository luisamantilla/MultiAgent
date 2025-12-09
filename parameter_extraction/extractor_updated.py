#this new file allows for folders to be inputted, and can take in pdfs, txt, or json files. 
from utils.vision_based_processor import VisionPDFExtractor
from pathlib import Path
import json, os, datetime, fitz

# === Configuration ===
input_path = Path("parameter_extraction/Tumor_tcell_papers_text")  # folder or single file
results_root = Path("parameter_extraction/results")
extractor = VisionPDFExtractor(model="gpt-4o", mode="parameters")

# === Helper: process a single PDF ===
def process_pdf(pdf_path: Path):
    """Extract parameters from a PDF page by page."""
    print(f"\n📘 Processing PDF: {pdf_path.name}")
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"❌ Failed to open PDF {pdf_path}: {e}")
        return
    total_pages = len(doc)
    doc.close()

    output_dir = results_root / f"{pdf_path.stem}_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    for page_index in range(total_pages):
        print(f"   ▶️ Page {page_index + 1}/{total_pages}")
        result = extractor.process_page(pdf_path, page_index=page_index, page_note=f"{pdf_path.name} - Page {page_index + 1}")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if result.get("success"):
            out_file = output_dir / f"{timestamp}_{pdf_path.stem}_page{page_index + 1}.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(result["content"], f, indent=2, ensure_ascii=False)
            print(f"      ✅ Saved: {out_file.name}")
        else:
            print(f"      ❌ Failed: {result.get('error')}")

# === Helper: process a text or JSON file ===
def process_text_or_json(file_path: Path):
    """Extract parameters from a text or JSON file using GPT (no vision)."""
    print(f"\n📝 Processing non-PDF file: {file_path.name}")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"❌ Failed to read {file_path}: {e}")
        return

    # Call GPT text mode (not vision)
    result = extractor.client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert at extracting structured data from scientific documents."},
            {"role": "user", "content": f"Extract all quantitative parameters and values from the following text:\n\n{content}"}
        ],
        temperature=0.1,
        max_tokens=3000
    )

    raw_output = result.choices[0].message.content
    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError:
        parsed = {"text_output": raw_output}  # fallback if not valid JSON

    # Save
    output_dir = results_root / f"{file_path.stem}_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{file_path.stem}_extracted.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(parsed, f, indent=2, ensure_ascii=False)
    print(f"      ✅ Saved: {out_file}")

# === Main logic ===
def process_input(path: Path):
    """Decide whether the input is a PDF, TXT, JSON, or folder."""
    if path.is_file():
        if path.suffix.lower() == ".pdf":
            process_pdf(path)
        elif path.suffix.lower() in [".txt", ".json"]:
            process_text_or_json(path)
        else:
            print(f"⚠️ Skipping unsupported file type: {path.name}")

    elif path.is_dir():
        files = sorted(list(path.glob("*")))
        print(f"📂 Found {len(files)} files in {path}")
        for file in files:
            if file.suffix.lower() == ".pdf":
                process_pdf(file)
            elif file.suffix.lower() in [".txt", ".json"]:
                process_text_or_json(file)
            else:
                print(f"⚠️ Skipping unsupported file type: {file.name}")
    else:
        print("❌ Invalid input. Please provide a valid file or folder.")

# === Run ===
process_input(input_path)
