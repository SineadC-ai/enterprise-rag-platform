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
import time

app = FastAPI(title="Enterprise RAG Platform", version="0.4.0")

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
    Retrieval-augmented chat:
    - Retrieve similar chunks
    - Build a concise context
    - Ask the LLM to answer using only that context
    """
    import time

    start_time = time.time()

    # Retrieval
    retrieved = search_similar_chunks(
        query=request.question,
        top_k=request.top_k,
        document_id=request.document_id,
    )

    # Limit how many chunks go into the prompt
    max_chunks_for_prompt = 8
    trimmed_chunks = retrieved[:max_chunks_for_prompt]

    # Build numbered context
    context_lines = []
    for idx, r in enumerate(trimmed_chunks):
        label = f"[{idx + 1}]"
        context_lines.append(
            f"{label} (doc={r['document_id']}, chunk={r['chunk_id']}): {r['text']}"
        )

    context_text = "\n\n".join(context_lines).strip()

    if context_text:
        prompt = (
            "You are a precise assistant that answers questions using only the context provided.\n"
            "If the answer is clearly stated in the context, answer it directly and concisely.\n"
            "If the context does not contain the answer, say: \"I don't know based on the provided context.\"\n"
            "Do not describe the context or limitations, just answer the question.\n\n"
            "Context passages:\n"
            f"{context_text}\n\n"
            "Question:\n"
            f"{request.question}\n\n"
            "Answer (do not add any extra sections or headings):"
        )
    else:
        prompt = (
            "There is no supporting context available for this question.\n"
            "If the question requires specific facts, say: \"I don't know based on the provided context.\"\n\n"
            f"Question:\n{request.question}\n\n"
            "Answer:"
        )

    answer = llm.chat(prompt=prompt, user_id=request.user_id)

    latency_ms = (time.time() - start_time) * 1000.0

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

