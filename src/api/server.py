from typing import Optional, List

from fastapi import Depends, FastAPI
from pydantic import BaseModel

from src.core.settings import get_settings, Settings
from src.llm.llm_client import LLMClient
from src.core.doc_store import save_document, list_documents
from src.core.pipeline import process_document, search_similar_chunks


app = FastAPI(title="Enterprise RAG Platform", version="0.4.0")


# ---------- Pydantic models ----------


class ChatRequest(BaseModel):
    question: str
    user_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str


class IngestRequest(BaseModel):
    document_id: str
    text: str
    source: Optional[str] = "unknown"


class IngestResponse(BaseModel):
    status: str
    document_id: str


class IngestAndEmbedRequest(BaseModel):
    document_id: str
    text: str
    source: Optional[str] = "unknown"
    chunk_size: int = 500
    overlap: int = 50


class IngestAndEmbedResponse(BaseModel):
    document_id: str
    num_chunks: int
    embedding_dim: int


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    document_id: Optional[str] = None


class SearchResult(BaseModel):
    document_id: str
    chunk_id: str
    chunk_index: int
    text: str
    score: float


class SearchResponse(BaseModel):
    results: List[SearchResult]


# ---------- Dependencies ----------


def get_llm_client() -> LLMClient:
    """
    Returns an LLM client instance used by the chat endpoint.
    The underlying implementation can be replaced without changing the route.
    """
    return LLMClient()


# ---------- Routes ----------


@app.get("/health")
def health(settings: Settings = Depends(get_settings)):
    """
    Basic health check endpoint exposing environment and region information.
    """
    return {
        "status": "ok",
        "environment": settings.environment,
        "region": settings.aws_region,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    llm: LLMClient = Depends(get_llm_client),
):
    """
    Chat endpoint that sends a question to the LLM client and returns a response.
    The current implementation uses a stubbed LLM client.
    """
    answer = llm.chat(prompt=request.question, user_id=request.user_id)
    return ChatResponse(answer=answer)


@app.post("/ingest", response_model=IngestResponse)
def ingest_document(request: IngestRequest):
    """
    Ingests a raw document and stores the original text with basic metadata.
    """
    source = request.source or "unknown"

    save_document(
        document_id=request.document_id,
        text=request.text,
        source=source,
    )

    return IngestResponse(
        status="document stored",
        document_id=request.document_id,
    )


@app.get("/documents")
def get_all_documents():
    """
    Returns all stored documents from the in-memory document store.
    Intended for local inspection and debugging.
    """
    return list_documents()


@app.post("/ingest_and_embed", response_model=IngestAndEmbedResponse)
def ingest_and_embed(request: IngestAndEmbedRequest):
    """
    Ingests a document, runs chunking and embedding, and persists embeddings if configured.
    Returns basic statistics about the processed document.
    """
    source = request.source or "unknown"

    save_document(
        document_id=request.document_id,
        text=request.text,
        source=source,
    )

    result = process_document(
        document_id=request.document_id,
        text=request.text,
        chunk_size=request.chunk_size,
        overlap=request.overlap,
    )

    return IngestAndEmbedResponse(
        document_id=result["document_id"],
        num_chunks=result["num_chunks"],
        embedding_dim=result["embedding_dim"],
    )


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    """
    Runs a vector similarity search over stored chunk embeddings and returns the most similar chunks.
    """
    results = search_similar_chunks(
        query=request.query,
        top_k=request.top_k,
        document_id=request.document_id,
    )

    api_results = [
        SearchResult(
            document_id=r["document_id"],
            chunk_id=r["chunk_id"],
            chunk_index=r["chunk_index"],
            text=r["text"],
            score=r["score"],
        )
        for r in results
    ]

    return SearchResponse(results=api_results)
