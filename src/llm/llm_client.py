from typing import Optional
import json
import boto3

from src.core.settings import get_settings


class LLMClient:
    """
    LLM client with two modes:
    - Stub mode (default)
    - Bedrock mode (when USE_BEDROCK_LLM=true)
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._use_bedrock = settings.use_bedrock_llm
        self._model_id = settings.bedrock_model_id
        self._region = settings.aws_region

        if self._use_bedrock:
            self._client = boto3.client(
                "bedrock-runtime",
                region_name=self._region
            )

    # -------------------------------
    # PUBLIC CHAT METHOD
    # -------------------------------
    def chat(self, prompt: str, user_id: Optional[str] = None) -> str:
        if self._use_bedrock:
            return self._call_bedrock(prompt)
        return self._call_stub(prompt)

    # -------------------------------
    # STUB IMPLEMENTATION
    # -------------------------------
    def _call_stub(self, prompt: str) -> str:
        preview = prompt.strip().replace("\n", " ")
        if len(preview) > 160:
            preview = preview[:157] + "..."
        return f"[stub-llm] {preview}"

    # -------------------------------
    # BEDROCK IMPLEMENTATION
    # -------------------------------
    # def _call_bedrock(self, prompt: str) -> str:
        # """
        # Bedrock call for Amazon Titan Text Lite.
        # """

        # body = {
            # "inputText": prompt,
            # "textGenerationConfig": {
                # "maxTokenCount": 512,
                # "temperature": 0.2,
                # "topP": 0.9
            # }
        # }

        # response = self._client.invoke_model(
            # modelId=self._model_id,              # should be amazon.titan-text-lite-v1
            # contentType="application/json",
            # accept="application/json",
            # body=json.dumps(body)
        # )

        # result = json.loads(response["body"].read())

        # Titan returns {"results": [{"outputText": "..."}]}
        #return result["results"][0]["outputText"]
        
    def _call_bedrock(self, prompt: str) -> str:
        """
        Bedrock call for Amazon Titan Text Lite.
        """

        # Minimal safety wrapping to avoid false-positive content filters
        safe_prompt = f"Provide a helpful, safe answer.\n\nUser: {prompt}\nAssistant:"

        body = {
            "inputText": safe_prompt,
            "textGenerationConfig": {
                "maxTokenCount": 512,
                "temperature": 0.2,
                "topP": 0.9
            }
        }

        response = self._client.invoke_model(
            modelId=self._model_id,              # amazon.titan-text-lite-v1 (or your chosen model)
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body)
        )

        result = json.loads(response["body"].read())

        # Titan returns {"results": [{"outputText": "..."}]}
        return result["results"][0]["outputText"]

    

    
    # def _call_bedrock(self, prompt: str) -> str:
        # """
        # Actual call to AWS Bedrock (Claude 3 Haiku by default).
        # """
        # body = {
            # "anthropic_version": "bedrock-2023-05-31",
            # "messages": [
                # {
                    # "role": "user",
                    # "content": prompt
                # }
            # ],
            # "max_tokens": 500
        # }

        # response = self._client.invoke_model(
            # modelId=self._model_id,
            # accept="application/json",
            # contentType="application/json",
            # body=json.dumps(body)
        # )

        # result = json.loads(response["body"].read())
        # return result["content"][0]["text"]
