import hashlib
import re

def clean_ocr_text(text: str) -> str:
    # Add spaces between camelCase or OCR joins
    text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)
    text = re.sub(r'(?<=[A-Za-z])(?=\d)', ' ', text)
    text = re.sub(r'(?<=\d)(?=[A-Za-z])', ' ', text)
    # Normalize whitespaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def get_text_hash(text: str, method='md5') -> str:
    text = text.strip().encode('utf-8')
    if method == 'sha256':
        return hashlib.sha256(text).hexdigest()
    return hashlib.md5(text).hexdigest()
