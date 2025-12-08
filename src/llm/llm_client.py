from typing import Optional
import json
import boto3
from src.core.settings import get_settings

class LLMClient:
    """
    LLM client used by the API layer.

    When USE_BEDROCK_LLM=false, calls are served by a simple stub for local development.
    When USE_BEDROCK_LLM=true, calls are sent to AWS Bedrock using the configured model.
    """

    def __init__(self) -> None:
        settings = get_settings()

        self._use_bedrock = settings.use_bedrock_llm
        self._model_id = settings.bedrock_model_id
        self._region = settings.aws_region

        # Optional Bedrock Guardrails configuration
        # These attributes are read from Settings so they can be controlled via .env
        self._guardrail_enabled = getattr(settings, "bedrock_guardrail_enabled", False)
        self._guardrail_id = getattr(settings, "bedrock_guardrail_id", None)
        self._guardrail_version = getattr(settings, "bedrock_guardrail_version", None)

        if self._use_bedrock:
            self._client = boto3.client(
                "bedrock-runtime",
                region_name=self._region,
            )
        else:
            self._client = None
    def chat(self, prompt: str, user_id: Optional[str] = None) -> str:
        """
        Returns a single-turn response for the given prompt.
        """
        if not self._use_bedrock:
            preview = prompt.replace("\n", " ")[:200]
            return f"(stub) LLM received: {preview}"
        return self._call_bedrock(prompt)

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
        Sends the prompt to the configured Bedrock text model.
        Guardrail configuration is attached when enabled.
        """
        body = {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": 512,
                "temperature": 0.2,
                "topP": 0.9,
            },
        }

        invoke_kwargs = {
            "modelId": self._model_id,
            "body": json.dumps(body),
            "contentType": "application/json",
            "accept": "application/json",
        }

        # Attach Bedrock Guardrails when enabled and configured
        if (
            self._guardrail_enabled
            and self._guardrail_id
            and self._guardrail_version
        ):
            # invoke_kwargs["guardrailConfig"] = {
                # "guardrailId": self._guardrail_id,
                # "guardrailVersion": self._guardrail_version,
            # }
            invoke_kwargs["guardrailIdentifier"] = self._guardrail_id
            invoke_kwargs["guardrailVersion"] = self._guardrail_version

        response = self._client.invoke_model(**invoke_kwargs)
        raw = response["body"].read().decode("utf-8")

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            # If response is not valid JSON, return the raw body
            return raw

        # Handle common Titan text response shapes
        # Examples:
        #   { "outputText": "..." }
        #   { "results": [ { "outputText": { "text": "..." } } ] }
        if isinstance(payload, dict):
            # Direct string output
            if isinstance(payload.get("outputText"), str):
                return payload["outputText"]

            # Results list with nested outputText
            results = payload.get("results")
            if isinstance(results, list) and results:
                first = results[0]
                if isinstance(first, dict):
                    output_text = first.get("outputText")
                    if isinstance(output_text, dict):
                        text = output_text.get("text")
                        if isinstance(text, str):
                            return text
                    text = first.get("text")
                    if isinstance(text, str):
                        return text
        # Fallback: return raw body string if structure is not recognized
        return raw
