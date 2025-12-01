from typing import Optional

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel

from src.core.settings import get_settings, Settings
from src.llm.llm_client import LLMClient
from src.core.doc_store import save_document, list_documents, get_document

app = FastAPI(title="Enterprise RAG Platform", version="0.2.0")


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
	
class DocumentResponse(BaseModel):
    document_id: str
    text: str
    source: str


def get_llm_client() -> LLMClient:
    return LLMClient()


@app.get("/health")
def health(settings: Settings = Depends(get_settings)):
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
    answer = llm.chat(prompt=request.question, user_id=request.user_id)
    return ChatResponse(answer=answer)


@app.post("/ingest", response_model=IngestResponse)
def ingest_document(request: IngestRequest):
    save_document(
        document_id=request.document_id,
        text=request.text,
        source=request.source or "unknown",
    )
    return IngestResponse(
        status="document stored",
        document_id=request.document_id,
    )

@app.get("/documents")
def get_all_docs():
    return list_documents()
	
@app.get("/documents/{document_id}", response_model=DocumentResponse)
def get_single_document(document_id: str):
    doc = get_document(document_id)

    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    return DocumentResponse(
        document_id=doc["document_id"],
        text=doc["text"],
        source=doc["source"],
    )


