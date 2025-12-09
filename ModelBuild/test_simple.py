# ask_param_bouchnita.py

import fitz  # PyMuPDF
from openai import OpenAI
import os

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Load the full text of bouchnita_2017.pdf manually
PDF_PATH = "parameter/my_papers/bouchnita_2017.pdf"

try:
    doc = fitz.open(PDF_PATH)
    full_text = "\n".join(page.get_text() for page in doc)
    doc.close()
except Exception as e:
    print(f"Failed to read PDF: {e}")
    exit()

# Define the parameter question
parameter_question = "What is the IFN-γ production rate for PD1-negative T cells?"
user_prompt = (
    f"You are reading a filtered section of a scientific paper. Focus on answering what is the refractory count threshold that triggers T cell exhaustion, even if inferred from context, simulations, or model assumptions. Your task is to answer:\n"
    f"{parameter_question}\n\n"
    "If the paper includes parameter values in tables, equations, or simulation initialization, prioritize those over inferred biological estimates."
    "If the value is not explicitly stated, infer a plausible value from figures, equations, simulation details, or common biological assumptions. "
    "Do not say 'not possible'—always give a value, even if inferred. "
    "Justify your answer, cite locations in the paper, and report a confidence score.\n"
    "Response format:\n"
    "Parameter: <value>\nUnits: <units>\nSource: <where you found it or inferred from>\nConfidence: <0.0 to 1.0>\n\n"
    f"{full_text}"
)

# Send to OpenAI GPT-4o
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a scientific research assistant who reads full PDFs and extracts parameter values with reasoning and calculating."},
        {"role": "user", "content": user_prompt}
    ],
    temperature=0.3
)

# Output the result
print("\n--- Extracted Parameter ---\n")
print(response.choices[0].message.content)
