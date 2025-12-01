import os
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import BaseModel

# Load variables from .env file into environment
load_dotenv()


class Settings(BaseModel):
    app_name: str = "Enterprise RAG Platform"
    environment: str = os.getenv("ENVIRONMENT", "local")
    aws_region: str = os.getenv("AWS_REGION", "us-east-1")
    bedrock_model_id: str = os.getenv(
        "BEDROCK_MODEL_ID",
        "stub-model",  # used until we wire real Bedrock
    )


@lru_cache
def get_settings() -> Settings:
    """
    Cached settings so we don't recreate on every request.
    """
    return Settings()
