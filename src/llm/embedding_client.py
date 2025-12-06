from typing import List, Dict
import hashlib
import json

import boto3

from src.core.settings import get_settings


class EmbeddingClient:
    """
    Embedding client with two modes:

    - Fake embeddings (default for local development)
    - Real embeddings via Amazon Titan (when USE_FAKE_EMBEDDINGS=false)

    Public interface used by the pipeline:

      - embed_text(text: str) -> List[float]
      - embed_texts(texts: List[str]) -> List[List[float]]
      - embed_chunks(chunks: List[Dict]) -> List[Dict]  # adds 'embedding' to each chunk
    """

    def __init__(self) -> None:
        settings = get_settings()

        self._use_fake = settings.use_fake_embeddings
        self._region = settings.aws_region
        self._model_id = settings.bedrock_embedding_model_id

        if not self._use_fake:
            self._client = boto3.client(
                "bedrock-runtime",
                region_name=self._region,
            )

    # -----------------------------
    # Public methods
    # -----------------------------

    def embed_text(self, text: str) -> List[float]:
        """
        Returns a single embedding vector for the given text.
        """
        if self._use_fake:
            return self._fake_embed(text)
        return self._bedrock_embed_single(text)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Returns embeddings for a list of texts.
        """
        if self._use_fake:
            return [self._fake_embed(t) for t in texts]
        return [self._bedrock_embed_single(t) for t in texts]

    def embed_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """
        Takes a list of chunk dictionaries and returns a new list
        where each chunk has an 'embedding' field added.

        Expects each chunk to have a 'text' key.
        """
        texts = [c["text"] for c in chunks]
        embeddings = self.embed_texts(texts)

        enriched: List[Dict] = []
        for chunk, emb in zip(chunks, embeddings):
            # create a shallow copy so we don't mutate the input list
            new_chunk = dict(chunk)
            new_chunk["embedding"] = emb
            enriched.append(new_chunk)

        return enriched

    # -----------------------------
    # Fake embeddings (dev mode)
    # -----------------------------

    def _fake_embed(self, text: str, dim: int = 16) -> List[float]:
        """
        Deterministic fake embedding used for local development.
        Produces a fixed-size vector based on a hash of the text.
        """
        h = hashlib.sha256(text.encode("utf-8")).digest()
        # Take the first `dim` bytes and normalize to [0, 1]
        return [b / 255.0 for b in h[:dim]]

    # -----------------------------
    # Bedrock Titan embeddings
    # -----------------------------

    def _bedrock_embed_single(self, text: str) -> List[float]:
        """
        Calls Amazon Titan Embeddings via Bedrock for a single text.
        Model is expected to be: amazon.titan-embed-text-v1
        """
        body = {
            "inputText": text
        }

        response = self._client.invoke_model(
            modelId=self._model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body),
        )

        result = json.loads(response["body"].read())

        # Titan embedding models return: { "embedding": [float, ...] }
        return result["embedding"]
