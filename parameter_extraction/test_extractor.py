from utils.vision_based_processor import VisionPDFExtractor
from pathlib import Path
import json, os, datetime, fitz

#updated on 10-08-25 to be able to receive folders and parse through documents in the folder
# === Configuration ===
input_path = Path("parameter_extraction/Tumor_tcell_papers_text")  # can be a single PDF or a folder
results_root = Path("parameter_extraction/results")

extractor = VisionPDFExtractor(model="gpt-4o", mode="parameters")

def process_pdf(pdf_path: Path):
    """Process all pages of a single PDF file."""
    print(f"\n📘 Processing file: {pdf_path.name}")
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    doc.close()

    for page_index in range(total_pages):
        print(f"   ▶️ Page {page_index + 1}/{total_pages}")
        result = extractor.process_page(pdf_path, page_index=page_index, page_note=f"{pdf_path.name} - Page {page_index + 1}")

        # Create output folder (timestamped for this run)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = results_root / f"{pdf_path.stem}_results"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save JSON for each page
        if result.get("success"):
            out_file = output_dir / f"{timestamp}_{pdf_path.stem}_page{page_index + 1}.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(result["content"], f, indent=2, ensure_ascii=False)
            print(f"      ✅ Saved: {out_file.name}")
        else:
            print(f"      ❌ Failed: {result.get('error')}")

# === Main logic ===
if input_path.is_file() and input_path.suffix.lower() == ".pdf":
    process_pdf(input_path)

elif input_path.is_dir():
    pdf_files = sorted(list(input_path.glob("*.pdf")))
    print(f"📂 Found {len(pdf_files)} PDF(s) in folder: {input_path}")
    for pdf_file in pdf_files:
        process_pdf(pdf_file)
else:
    print("❌ Invalid input. Please provide a valid PDF file or folder path.")
