import os
import logging
from typing import List
from datetime import datetime
import pytesseract
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
from concurrent.futures import ThreadPoolExecutor, as_completed
from docx import Document as DocxDocument
from rag.utils import clean_ocr_text, get_text_hash

# Wrapper for document content + metadata
class DocumentWrapper:
    def __init__(self, page_content: str, metadata: dict):
        self.page_content = page_content
        self.metadata = metadata

# Setup logging
def setup_logger():
    logger = logging.getLogger("doc_loader")
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

logger = setup_logger()

# OCR a single page
def ocr_page(img) -> str:
    return pytesseract.image_to_string(img)

# Run OCR on scanned PDFs
def ocr_pdf(path: str, max_workers: int = 4) -> str:
    try:
        images = convert_from_path(path)
        texts = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(ocr_page, img) for img in images]
            for future in as_completed(futures):
                try:
                    page_text = future.result()
                    texts.append(page_text)
                except Exception as e:
                    logger.warning(f"OCR page failed: {e}")
        return "\n".join(texts).strip()
    except Exception as e:
        logger.warning(f"OCR failed on {os.path.basename(path)}: {e}")
        return ""

# Extract text from readable PDFs
def extract_text_from_pdf(path: str) -> str:
    try:
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text.strip()
    except Exception as e:
        logger.warning(f"Failed to extract text from PDF '{os.path.basename(path)}': {e}")
        return ""

# Extract text from Word documents
def extract_text_from_docx(path: str) -> str:
    try:
        doc = DocxDocument(path)
        return "\n".join(para.text for para in doc.paragraphs).strip()
    except Exception as e:
        logger.warning(f"Failed to read .docx file '{os.path.basename(path)}': {e}")
        return ""

# Load and extract text from files in a folder
def load_text_from_scanned_pdfs(pdf_dir: str) -> List[DocumentWrapper]:
    """Loads PDFs, DOCX, TXT, and MD files and extracts text, including OCR fallback for scanned PDFs."""
    loaded_docs = []
    seen_hashes = set()
    supported_extensions = {".pdf", ".txt", ".md", ".docx"}

    all_files = [f for f in os.listdir(pdf_dir)
                 if os.path.splitext(f)[1].lower() in supported_extensions]
    logger.info(f"Found {len(all_files)} supported files: {all_files}")

    for file in all_files:
        path = os.path.join(pdf_dir, file)
        logger.info(f"Processing: {file}")

        ext = os.path.splitext(file)[1].lower()
        text = ""

        if ext in [".txt", ".md"]:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
            except Exception as e:
                logger.warning(f"Failed to read {ext} file '{file}': {e}")
                continue
        elif ext == ".docx":
            text = extract_text_from_docx(path)
        elif ext == ".pdf":
            text = extract_text_from_pdf(path)
            if not text:
                logger.info("No extractable text found in PDF, running OCR...")
                text = ocr_pdf(path)

        if not text:
            logger.info(f"Skipping {file} (no text extracted).")
            continue

        # Clean, deduplicate
        text = clean_ocr_text(text)
        text_hash = get_text_hash(text)
        if text_hash in seen_hashes:
            logger.info(f"Duplicate skipped: {file}")
            continue
        seen_hashes.add(text_hash)

        loaded_docs.append(
            DocumentWrapper(page_content=text, metadata={
                "source": file,
                "hash": text_hash,
                "loaded_at": datetime.now().isoformat()
            })
        )
        logger.info(f"Loaded '{file}' with hash: {text_hash[:8]}...")

    return loaded_docs
