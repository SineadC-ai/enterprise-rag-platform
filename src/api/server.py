from typing import Optional, List
from fastapi import Depends, FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from src.core.settings import get_settings, Settings
from src.llm.llm_client import LLMClient
from src.core.doc_store import save_document, list_documents
from src.core.pipeline import process_document, search_similar_chunks
from src.core.file_loader import load_text_from_bytes
from src.rag.vector_store import PgVectorStore
from fastapi.middleware.cors import CORSMiddleware
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
import time
import logging
from src.core.logging_config import setup_logging
from src.core.otel_config import setup_tracing

setup_logging()
#setup_tracing() #will display on bash console for now. 
app = FastAPI(title="Enterprise RAG Platform", version="0.4.0")

FastAPIInstrumentor.instrument_app(app)


# --- CORS (needed for browser frontends) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # for local dev; can restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



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

class RagChatRequest(BaseModel):
    question: str
    top_k: int = 5
    document_id: Optional[str] = None
    user_id: Optional[str] = None

class RagChatResponse(BaseModel):
    answer: str
    context: List[SearchResult]
	
class DebugConfig(BaseModel):
    environment: str
    aws_region: str
    llm_model_id: str
    use_bedrock_llm: bool
    use_fake_embeddings: bool
    embedding_model_id: str
    persist_embeddings: bool
    database_url_prefix: str

class RagHealth(BaseModel):
    status: str
    num_chunks: int
    num_documents: int
    use_fake_embeddings: bool
    use_bedrock_llm: bool

class DebugChunk(BaseModel):
    document_id: str
    chunk_id: str
    chunk_index: int
    text_preview: str
    text_length: int
    embedding_dim: int | None = None


class DebugDocumentChunksResponse(BaseModel):
    document_id: str
    limit: int
    total_chunks_returned: int
    chunks: List[DebugChunk]


class DebugSearchRequest(BaseModel):
    query: str
    top_k: int = 5
    document_id: Optional[str] = None


class DebugSearchResponse(BaseModel):
    query: str
    top_k: int
    document_id: Optional[str] = None
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

@app.post("/ingest_file", response_model=IngestAndEmbedResponse)
async def ingest_file(
    file: UploadFile = File(...),
    chunk_size: int = 500,
    overlap: int = 50,
):
    """
    Ingests a text file, runs chunking and embeddings, and stores
    the processed chunks in the vector store.
    """
    content = await file.read()
    text, media_type = load_text_from_bytes(file.filename, content)

    if not text.strip():
        raise HTTPException(
            status_code=400,
            detail="File contained no readable text or unsupported format.",
        )

    document_id = file.filename

    save_document(
        document_id=document_id,
        text=text,
        source=media_type,
    )

    result = process_document(
        document_id=document_id,
        text=text,
        chunk_size=chunk_size,
        overlap=overlap,
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


@app.post("/chat_rag", response_model=RagChatResponse)
def chat_rag(
    request: RagChatRequest,
    llm: LLMClient = Depends(get_llm_client),
):
    """
    RAG-style chat:
    1) Retrieve similar chunks
    2) Build context
    3) Call LLM with context + question

    This version also logs simple timing information for retrieval and LLM.
    """

    t_start = time.time()

    # 1) Retrieval
    t_retrieval_start = time.time()
    retrieved = search_similar_chunks(
        query=request.question,
        top_k=request.top_k,
        document_id=request.document_id,
    )
    t_retrieval_end = time.time()

    # Guardrail: no context found in vector store
    if not retrieved:
        t_llm_start = time.time()
        fallback_prompt = (
            "No relevant context was found in the knowledge base.\n\n"
            f"User question: {request.question}\n\n"
            "Answer based on general knowledge. "
            "If you do not know the answer, say that you do not know."
        )
        answer = llm.chat(prompt=fallback_prompt, user_id=request.user_id)
        t_llm_end = time.time()
        t_end = time.time()

        # Simple timing log
        print(
            "[chat_rag] no-context "
            f"total={t_end - t_start:.3f}s "
            f"retrieval={t_retrieval_end - t_retrieval_start:.3f}s "
            f"llm={t_llm_end - t_llm_start:.3f}s"
        )

        return RagChatResponse(answer=answer, context=[])

    # 2) Build context text from retrieved chunks
    context_text = "\n\n".join(
        f"[{r['document_id']} / {r['chunk_id']}] {r['text']}"
        for r in retrieved
    )

    prompt = (
        "Context:\n"
        f"{context_text}\n\n"
        "Question:\n"
        f"{request.question}\n\n"
        "Answer the question using only the information in the context above. "
        "If the context does not contain the answer, say that the information "
        "is not available."
    )

    # 3) LLM call
    t_llm_start = time.time()
    answer = llm.chat(prompt=prompt, user_id=request.user_id)
    t_llm_end = time.time()
    t_end = time.time()

    # Simple timing log
    print(
        "[chat_rag] ok "
        f"total={t_end - t_start:.3f}s "
        f"retrieval={t_retrieval_end - t_retrieval_start:.3f}s "
        f"llm={t_llm_end - t_llm_start:.3f}s"
    )

    api_context = [
        SearchResult(
            document_id=r["document_id"],
            chunk_id=r["chunk_id"],
            chunk_index=r["chunk_index"],
            text=r["text"],
            score=r["score"],
        )
        for r in retrieved
    ]

    return RagChatResponse(
        answer=answer,
        context=api_context,
    )


@app.get("/debug/config", response_model=DebugConfig)
def debug_config(settings: Settings = Depends(get_settings)):
    """
    Returns the current configuration flags and key settings.
    Useful for local inspection and demos.
    """
    # Avoid leaking full DB URL in API response
    db_url = settings.database_url or ""
    if "@" in db_url:
        db_url_prefix = db_url.split("@", 1)[1][:60]
    else:
        db_url_prefix = db_url[:60]

    return DebugConfig(
        environment=settings.environment,
        aws_region=settings.aws_region,
        llm_model_id=settings.bedrock_model_id,
        use_bedrock_llm=settings.use_bedrock_llm,
        use_fake_embeddings=settings.use_fake_embeddings,
        embedding_model_id=settings.bedrock_embedding_model_id,
        persist_embeddings=settings.persist_embeddings,
        database_url_prefix=db_url_prefix,
    )

@app.get("/debug/health_rag", response_model=RagHealth)
def debug_health_rag(settings: Settings = Depends(get_settings)):
    """
    Runs a lightweight health check against the vector store.
    Returns counts of documents/chunks plus mode flags.
    """
    store = PgVectorStore()
    stats = store.get_stats()

    status = "ok" if stats["num_chunks"] > 0 else "empty"

    return RagHealth(
        status=status,
        num_chunks=stats["num_chunks"],
        num_documents=stats["num_documents"],
        use_fake_embeddings=settings.use_fake_embeddings,
        use_bedrock_llm=settings.use_bedrock_llm,
    )

@app.get("/debug/document_chunks/{document_id}", response_model=DebugDocumentChunksResponse)
def debug_document_chunks(document_id: str, limit: int = 200):
    """
    Return stored chunks for a single document for debugging.
    Helps verify chunking, ordering, and embedding dimensions.
    """
    store = PgVectorStore()
    rows = store.get_chunks_for_document(document_id=document_id, limit=limit)

    debug_chunks = [
        DebugChunk(
            document_id=row["document_id"],
            chunk_id=row["chunk_id"],
            chunk_index=row["chunk_index"],
            text_preview=row["text"][:200],
            text_length=row["text_length"],
            embedding_dim=row["embedding_dim"],
        )
        for row in rows
    ]

    return DebugDocumentChunksResponse(
        document_id=document_id,
        limit=limit,
        total_chunks_returned=len(debug_chunks),
        chunks=debug_chunks,
    )

@app.post("/debug/search_raw", response_model=DebugSearchResponse)
def debug_search_raw(request: DebugSearchRequest):
    """
    Run a search and return the same structure as /search, plus echo query + filters.
    Useful for debugging retrieval behavior.
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

    return DebugSearchResponse(
        query=request.query,
        top_k=request.top_k,
        document_id=request.document_id,
        results=api_results,
    )
