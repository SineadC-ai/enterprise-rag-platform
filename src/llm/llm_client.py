from typing import Optional

from src.core.settings import get_settings


class LLMClient:
    def __init__(self) -> None:
        settings = get_settings()
        self.model_id = settings.bedrock_model_id
        self.region = settings.aws_region

    def chat(self, prompt: str, user_id: Optional[str] = None) -> str:
        return (
            f"(stubbed response) Model '{self.model_id}' in region '{self.region}' "
            f"would answer user '{user_id}' question:\n\n{prompt}"
        )
