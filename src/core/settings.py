import os
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import List, Dict, Optional

# Load environment variables from .env at project root
load_dotenv()

class Settings(BaseModel):
    """
    Application configuration loaded from environment variables.
    """

    app_name: str = "Enterprise RAG Platform"
    environment: str = os.getenv("ENVIRONMENT", "local")
    aws_region: str = os.getenv("AWS_REGION", "us-east-1")

    # ---------- LLM configs ----------

    # Main chat model (Claude via Bedrock)
    bedrock_model_id: str = os.getenv(
        "BEDROCK_MODEL_ID",
        "anthropic.claude-3-haiku-20240307-v1:0",
    )

    # Embedding model (for real embeddings in a later version)
    bedrock_embedding_model_id: str = os.getenv(
        "BEDROCK_EMBEDDING_MODEL_ID",
        "amazon.titan-embed-text-v1",
    )

    # Use fake embeddings for local development
    use_fake_embeddings: bool = (
        os.getenv("USE_FAKE_EMBEDDINGS", "true").lower() == "true"
    )

    # Toggle real Bedrock LLM calls vs stub implementation
    use_bedrock_llm: bool = (
        os.getenv("USE_BEDROCK_LLM", "false").lower() == "true"
    )
    #Bedrock Guardrails configuration
    bedrock_guardrail_id: Optional[str] = os.getenv("BEDROCK_GUARDRAIL_ID")
    bedrock_guardrail_version: Optional[str] = os.getenv("BEDROCK_GUARDRAIL_VERSION")
    bedrock_guardrail_enabled: bool = os.getenv("BEDROCK_GUARDRAIL_ENABLED", "false" ).lower() == "true"

    # ---------- Data / Vector store ----------

    # Persist chunk embeddings to Postgres / Neon
    persist_embeddings: bool = (
        os.getenv("PERSIST_EMBEDDINGS", "true").lower() == "true"
    )

    # Postgres / Neon connection string
    database_url: str = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:postgres@localhost:5432/rag_platform",
    )


def get_settings() -> Settings:
    """
    Returns a settings instance.
    """
    return Settings()
