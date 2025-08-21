import os
import re
import spacy
import logging
from rag.doc_loader import load_text_from_scanned_pdfs, DocumentWrapper
from rag.embed_pipeline import build_vector_db

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Parameters
pdf_dir = "./data/pdfs"
MAX_CHARS = 1500

logging.info(f"Scanning folder: {os.path.abspath(pdf_dir)}")
documents = load_text_from_scanned_pdfs(pdf_dir)
logging.info(f"Total documents loaded: {len(documents)}")


def clean_ocr_text(text: str) -> str:
    text = re.sub(r'\n{2,}', '\n\n', text)  # Preserve paragraphs
    text = re.sub(r'[^\S\r\n]+', ' ', text)  # Normalize spaces
    text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)  # Fix hyphenated line breaks
    return text.strip()


def is_table_like(text: str) -> bool:
    lines = text.strip().splitlines()
    return len(lines) > 3 and all(re.search(r'\d', line) for line in lines)


def split_by_headings(text):
    heading_pattern = re.compile(
        r'\n(?=\d+(\.\d+)*\s+[A-Z])'  # Numbered headings
        r'|\n(?=[A-Z][A-Z\s\-]{5,})'  # ALL CAPS headings
    )
    splits = heading_pattern.split(text)
    return [s.strip() for s in splits if s.strip()]


def chunk_section(section, max_chars=MAX_CHARS):
    if is_table_like(section):
        return [section.strip()]

    doc = nlp(section)
    sentences = [sent.text.strip() for sent in doc.sents]
    chunks, current = [], ""

    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= max_chars:
            current = f"{current} {sentence}".strip() if current else sentence
        else:
            if current:
                chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    return chunks


def smart_chunk(text, max_chars=MAX_CHARS):
    cleaned = clean_ocr_text(text)
    sections = split_by_headings(cleaned)
    all_chunks = []
    for section in sections:
        all_chunks.extend(chunk_section(section, max_chars))
    return all_chunks


# Process documents
all_chunks, all_metadata = [], []

for doc in documents:
    chunks = smart_chunk(doc.page_content)
    metadata = doc.metadata.copy()
    for i, chunk in enumerate(chunks):
        all_chunks.append(chunk)
        enriched_meta = {
            **metadata,
            "chunk_index": i,
            "source_file": os.path.basename(metadata.get("source", "unknown"))
        }
        all_metadata.append(enriched_meta)

logging.info(f"Total text chunks prepared: {len(all_chunks)}")

# Preview chunks
for i, chunk in enumerate(all_chunks[:5]):
    logging.info(f"--- Chunk {i + 1} ---\n{chunk[:500]}")

# Build vector DB
build_vector_db(documents=all_chunks, metadatas=all_metadata)

logging.info("RAG pipeline completed successfully.")
