# src/core/file_loader.py

from typing import Tuple
from io import BytesIO

from pypdf import PdfReader


def load_text_from_bytes(filename: str, content: bytes) -> Tuple[str, str]:
    """
    Converts file bytes into text.

    Supports:
      - .txt  -> UTF-8 decoded text
      - .pdf  -> extracted text via pypdf

    Returns:
      (text, media_type)
    """
    name_lower = filename.lower()

    if name_lower.endswith(".txt"):
        text = content.decode("utf-8", errors="ignore")
        media_type = "text/plain"
		
    elif name_lower.endswith(".py"):
        text = content.decode("utf-8", errors="ignore")
        media_type = "text/plain"

    elif name_lower.endswith(".pdf"):
        text = _extract_text_from_pdf(content)
        media_type = "application/pdf"

    else:
        # Fallback: unsupported type for now
        text = ""
        media_type = "application/octet-stream"

    return text, media_type


def _extract_text_from_pdf(content: bytes) -> str:
    """
    Extracts text from a PDF file using pypdf.
    """
    reader = PdfReader(BytesIO(content))
    parts = []

    for page in reader.pages:
        page_text = page.extract_text() or ""
        parts.append(page_text)

    return "\n\n".join(parts)
