from typing import List, Dict


DEFAULT_CHUNK_SIZE = 500      # characters
DEFAULT_CHUNK_OVERLAP = 50    # characters


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    document_id: str = ""
) -> List[Dict]:
    """
    Split text into overlapping chunks.
    Returns a list of dictionaries with metadata.
    """

    chunks = []
    start = 0
    idx = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text_value = text[start:end]

        chunks.append({
            "chunk_id": f"{document_id}_chunk_{idx}",
            "document_id": document_id,
            "index": idx,
            "text": chunk_text_value.strip(),
        })

        idx += 1
        start += (chunk_size - overlap)

    return chunks
