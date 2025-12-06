from typing import List, Dict, Optional

from src.core.settings import get_settings
from src.llm.embedding_client import EmbeddingClient
from src.rag.vector_store import PgVectorStore
from src.rag.chunking import (
    chunk_text,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
)


def chunk_and_embed_document(
    document_id: str,
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    source: Optional[str] = None,
    user_id: Optional[str] = None,
    filename: Optional[str] = None,
) -> List[Dict]:
    """
    Split a document into chunks and generate embeddings for each chunk.

    Returns a list of dictionaries, one per chunk, with:
      - document_id
      - chunk_id
      - chunk_index
      - text
      - embedding
      - source (optional)
      - user_id (optional)
      - filename (optional)
    """
    # 1) Chunk the raw text
    chunks = chunk_text(
        text=text,
        chunk_size=chunk_size,
        overlap=overlap,
        document_id=document_id,
    )

    # 2) Generate embeddings
    embedding_client = EmbeddingClient()
    texts = [c["text"] for c in chunks]
    embeddings = embedding_client.embed_texts(texts)

    enriched: List[Dict] = []
    for chunk, embedding in zip(chunks, embeddings):
        enriched.append(
            {
                "document_id": chunk["document_id"],
                "chunk_id": chunk["chunk_id"],
                "chunk_index": chunk["index"],
                "text": chunk["text"],
                "embedding": embedding,
                # metadata fields (may be None)
                "source": source,
                "user_id": user_id,
                "filename": filename,
            }
        )

    return enriched


def process_document(
    document_id: str,
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    source: Optional[str] = None,
    user_id: Optional[str] = None,
    filename: Optional[str] = None,
) -> Dict:
    """
    High-level document processing pipeline.

    Steps:
      1. Chunk the document.
      2. Generate embeddings for each chunk.
      3. Optionally persist embeddings to the vector store (pgvector).

    Returns a summary dictionary with:
      - document_id
      - num_chunks
      - embedding_dim
    """
    settings = get_settings()

    # Chunk + embed
    enriched_chunks = chunk_and_embed_document(
        document_id=document_id,
        text=text,
        chunk_size=chunk_size,
        overlap=overlap,
        source=source,
        user_id=user_id,
        filename=filename,
    )

    # Persist embeddings if enabled
    if settings.persist_embeddings:
        store = PgVectorStore()
        store.upsert_embeddings(enriched_chunks)

    embedding_dim = 0
    if enriched_chunks and "embedding" in enriched_chunks[0]:
        embedding_dim = len(enriched_chunks[0]["embedding"])

    return {
        "document_id": document_id,
        "num_chunks": len(enriched_chunks),
        "embedding_dim": embedding_dim,
    }


def search_similar_chunks(
    query: str,
    top_k: int = 5,
    document_id: Optional[str] = None,
) -> List[Dict]:
    """
    Runs a semantic similarity search over stored chunk embeddings.

    - Encodes the query into an embedding.
    - Uses the vector store to perform similarity search.
    - Returns a list of result dictionaries with:
        document_id, chunk_id, chunk_index, text, score
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
