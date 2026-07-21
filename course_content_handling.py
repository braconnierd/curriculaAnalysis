# ------------------------------------------------------------------------------
# Import and Library Management
# ------------------------------------------------------------------------------
import ollama
from pathlib import Path
from ollama_utils import ensure_ollama_running, stop_ollama
from pdf2image import convert_from_path
import tempfile
import pdfplumber
import os

# -----------------------------------
# Script Parameters
# -----------------------------------
input_folder = Path("")       # where PDFs are
output_folder = Path("")     # where TXT files go
output_folder.mkdir(parents=True, exist_ok=True)
llm_transcription_model="qwen2.5vl:3b" # model used to the handwriting transcription

# -----------------------------
# LLM Transcription Functions
# -----------------------------
def transcribe_image(image_path):
    """Use a visual LLM to transcribe and structure handwritten notes."""

    prompt = (
        "Transcribe the handwritten page faithfully."
        "Rules:"
        "- Preserve all content and its order."
        "- Do NOT summarize or omit material."
        "- Do NOT invent content not visible in the image."
        "- When mathematical notation appears, transcribe it into clear words instead of symbols."
        "- Keep equations as bracketed placeholders, e.g. [Equation: description]."
        "- Preserve paragraph and line structure where possible."
        "- If a word or symbol is unreadable, write [UNCLEAR]."

        "Goal: produce a complete, readable transcription of everything on the page."
        )

    response = ollama.generate(
        model=llm_transcription_model,
        
        options={
           "temperature": 0,
            "num_predict": 800,
            "num_ctx": 24000,
            
            "repeat_penalty": 1.85,
            "presence_penalty": 1.1,
            "frequency_penalty": 1.1,

            "top_p": 0.9,
        },
        prompt=prompt,
        images=[str(image_path)],
    )

    return response["response"].strip()

def transcribe_pdf_as_image(pdf_path):

    """Convert each page of a PDF to an image and transcribe it."""
    pdf_path = Path(pdf_path).resolve()
    print(f"Processing: {pdf_path.name}")

    # Convert PDF pages to temporary images
    pages = convert_from_path(pdf_path, dpi=200)
    
    print("Loaded", len(pages), "pages")
    all_text = []

    # Loop over all the pages and transcribe each one
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, page in enumerate(pages):
            image_file = Path(tmpdir) / f"page_{i+1}.png"
            page.save(image_file, "PNG")
            print(f"  -> Transcribing page {i+1}/{len(pages)}...")
            text = transcribe_image(image_file)
            all_text.append(f"\n\n--- Page {i+1} ---\n{text}")

    return "\n".join(all_text)

# -----------------------------
# PDF Transcription Function
# -----------------------------
def extract_text_from_pdf(pdf_path):
    """Use pdfplumber to extract all the text from the pdf"""
    text_chunks = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                try:
                # Extract text
                    text = page.extract_text()
                    if text:
                        text_chunks.append(text)
                except Exception:
                    continue
    except Exception as e:
        print(f"[ERROR] Failed to open PDF: {pdf_path} — {e}")
        return ""
                    
    return "\n".join(text_chunks).strip()

# -----------------------------
# Key Functions
# -----------------------------
def process_course_content(input_root: str, output_root: str):
    """
    Mirrors folder structure in output_root, converting all files to .txt.
    Only uses LLM when PDFs contain no extractable text.
    """
    input_root = Path(input_root)
    output_root = Path(output_root)

    print(f"\n📁 Processing course content from: {input_root}")
    print(f"📁 Output will be saved to: {output_root}\n")

    for folder, subdirs, files in os.walk(input_root):

        # Define all the relevant paths
        folder_path = Path(folder)
        rel_path = folder_path.relative_to(input_root)  
        output_folder = output_root / rel_path

        # Make sure output folder exists
        output_folder.mkdir(parents=True, exist_ok=True)


        print(f"\n=== Folder: {rel_path} ===")

        for file in files:
            input_file_path = folder_path / file

            # Determine output .txt file path
            base_name = file.rsplit(".", 1)[0]  # remove extension
            output_txt_path = output_folder / f"{base_name}.txt"

            print(f"- Converting {file} → {output_txt_path.relative_to(output_root)}")

            convert_file_to_txt(input_file_path, output_txt_path)

    print("\n All files processed.\n")

def convert_file_to_txt(input_path: Path, output_path: Path) -> None:
    """
    Converts a single file into a .txt file and saves to output_path.
    Handles pdf → text with fallback.
    """

    suffix = input_path.suffix.lower()

    if suffix in [".txt", ".md"]:
        # Direct text files → just copy content
        try:
            text = input_path.read_text(errors="ignore")
        except Exception as e:
            print(f"[ERROR reading {input_path}: {e}]")
            text = ""
        output_path.write_text(text, encoding="utf-8")
        return

    elif suffix == ".pdf":
        print(f"   → Extracting PDF: {input_path.name}")

        # Try pdfplumber
        extracted = extract_text_from_pdf(input_path)

        if extracted.strip():
            print("     ✓ Extracted using pdfplumber")
            output_path.write_text(extracted, encoding="utf-8")
            return
        else:
            print("     ⚠ No text found — using LLM fallback (transcribe_pdf_as_image)")
            fallback_text = transcribe_pdf_as_image(str(input_path))
            output_path.write_text(fallback_text, encoding="utf-8")
            return

    else:
        print(f"   → Skipping unsupported file: {input_path.name}")
        return

# -----------------------------
# Main Loop
# -----------------------------


# Turn Ollama on 
ensure_ollama_running()

process_course_content(input_folder, output_folder)

#Turn Ollama off
stop_ollama()
