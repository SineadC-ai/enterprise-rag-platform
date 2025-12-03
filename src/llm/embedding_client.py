from typing import List, Dict, Any
import hashlib

from src.core.settings import get_settings


class EmbeddingClient:
    """
    Embedding client that supports:
    - Fake embeddings (default, for local dev and architecture work)
    - Real Bedrock embeddings (later, when USE_FAKE_EMBEDDINGS=false)

    Right now we only implement fake mode. The structure is ready for Bedrock.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self.region = settings.aws_region
        self.model_id = settings.bedrock_embedding_model_id
        self.use_fake = settings.use_fake_embeddings

        # Placeholder: when we turn on real Bedrock, we will initialize boto3 here.
        # For now, we don't import boto3 to avoid credentials noise in fake mode.

    def _fake_embedding(self, text: str, dim: int = 16) -> List[float]:
        """
        Deterministic fake embedding:
        - Uses SHA-256 hash of the text
        - Maps first `dim` bytes into floats in [-1, 1]
        This is enough to exercise the pipeline and test vector DB wiring later.
        """
        h = hashlib.sha256(text.encode("utf-8")).digest()
        # Take the first `dim` bytes and map each to [-1, 1]
        vec = []
        for b in h[:dim]:
            # 0..255 -> -1..1
            vec.append((b / 127.5) - 1.0)
        return vec

    def embed_text(self, text: str) -> List[float]:
        """
        Return an embedding vector for a single text.
        Currently fake-only (no AWS calls).
        """
        if self.use_fake:
            return self._fake_embedding(text)

        # Placeholder for future real Bedrock call
        raise NotImplementedError(
            "Real Bedrock embeddings not implemented yet. "
            "Set USE_FAKE_EMBEDDINGS=true in your .env."
        )

    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Take a list of chunk dicts (with 'text') and return a list where each
        chunk is enriched with an 'embedding' field (list[float]).
        """
        enriched = []
        for chunk in chunks:
            text = chunk["text"]
            vector = self.embed_text(text)
            enriched.append(
                {
                    **chunk,
                    "embedding": vector,
                }
            )
        return enriched
