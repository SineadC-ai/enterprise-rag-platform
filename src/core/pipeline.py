from typing import List, Dict, Any

from src.rag.chunking import chunk_text
from src.llm.embedding_client import EmbeddingClient
from src.rag.vector_store import PgVectorStore
from src.core.settings import get_settings


def chunk_and_embed_document(
    document_id: str,
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> List[Dict[str, Any]]:
    """
    Chunks a document and generates embeddings for each chunk.
    Returns a list of chunk dictionaries with an 'embedding' field.
    """
    chunks = chunk_text(
        text=text,
        chunk_size=chunk_size,
        overlap=overlap,
        document_id=document_id,
    )

    embedding_client = EmbeddingClient()
    enriched_chunks = embedding_client.embed_chunks(chunks)
    return enriched_chunks


def process_document(
    document_id: str,
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> Dict[str, Any]:
    """
    Runs chunking and embedding for a document and persists embeddings if configured.
    Returns basic statistics about the processed document.
    """
    settings = get_settings()

    enriched_chunks = chunk_and_embed_document(
        document_id=document_id,
        text=text,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    if settings.persist_embeddings and enriched_chunks:
        store = PgVectorStore()
        store.upsert_embeddings(enriched_chunks)

    num_chunks = len(enriched_chunks)
    embedding_dim = len(enriched_chunks[0]["embedding"]) if num_chunks > 0 else 0

    return {
        "document_id": document_id,
        "num_chunks": num_chunks,
        "embedding_dim": embedding_dim,
    }


def search_similar_chunks(
    query: str,
    top_k: int = 5,
    document_id: str | None = None,
) -> List[Dict[str, Any]]:
    """
    Embeds a query string and searches for similar chunks in the vector store.
    """
    embedding_client = EmbeddingClient()
    query_embedding = embedding_client.embed_text(query)

    store = PgVectorStore()
    results = store.search_similar_chunks(
        query_embedding=query_embedding,
        top_k=top_k,
        document_id=document_id,
    )
    return results
