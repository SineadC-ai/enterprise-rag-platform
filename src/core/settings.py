import os
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


class Settings(BaseModel):
    app_name: str = "Enterprise RAG Platform"
    environment: str = os.getenv("ENVIRONMENT", "local")
    aws_region: str = os.getenv("AWS_REGION", "us-east-1")

    bedrock_model_id: str = os.getenv(
        "BEDROCK_MODEL_ID",
        "anthropic.claude-3-haiku-20240307-v1:0",
    )

    bedrock_embedding_model_id: str = os.getenv(
        "BEDROCK_EMBEDDING_MODEL_ID",
        "amazon.titan-embed-text-v1",
    )

    use_fake_embeddings: bool = os.getenv("USE_FAKE_EMBEDDINGS", "true").lower() == "true"
    # Controls whether embeddings are written to the database
    persist_embeddings: bool = os.getenv("PERSIST_EMBEDDINGS", "true").lower() == "true"
	
    # Postgres database URL, used when persisting embeddings with pgvector
    database_url: str = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:postgres@localhost:5432/rag_platform",
    )




@lru_cache
def get_settings() -> Settings:
    return Settings()
