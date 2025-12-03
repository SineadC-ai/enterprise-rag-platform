from typing import Dict, Any, Optional

# Simple in-memory document store (temporary)
DOCUMENT_STORE: Dict[str, Dict[str, Any]] = {}


def save_document(document_id: str, text: str, source: str) -> None:
    """
    Store a raw document in the in-memory store.
    Later this will be replaced by a real database + S3.
    """
    DOCUMENT_STORE[document_id] = {
        "document_id": document_id,
        "text": text,
        "source": source,
    }


def get_document(document_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a stored document (used later for chunking)."""
    return DOCUMENT_STORE.get(document_id)


def list_documents() -> Dict[str, Dict[str, Any]]:
    """Return all stored documents (for debugging / inspection)."""
    return DOCUMENT_STORE
