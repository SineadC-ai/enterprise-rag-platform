from typing import Optional

from fastapi import Depends, FastAPI
from pydantic import BaseModel

from src.core.settings import get_settings, Settings
from src.llm.llm_client import LLMClient

app = FastAPI(title="Enterprise RAG Platform", version="0.2.0")


class ChatRequest(BaseModel):
    question: str
    user_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str


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
