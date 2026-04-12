"""OpenAI embedder for query strings."""

import os
from typing import Optional

from openai import AsyncOpenAI


class Embedder:
    """Generates embeddings via OpenAI. Injected into clients for testability."""

    def __init__(self, model: str, api_key: Optional[str] = None) -> None:
        """Initialize embedder.

        Args:
            model: OpenAI embedding model name (e.g. text-embedding-3-small)
            api_key: Optional key. Falls back to OPENAI_API_KEY env var at embed time.
        """
        self.model = model
        self._api_key = api_key
        self._client: Optional[AsyncOpenAI] = None

    def _get_client(self) -> AsyncOpenAI:
        # Reason: lazy init so construction doesn't require OPENAI_API_KEY,
        # which keeps server setup testable without secrets.
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self._api_key or os.getenv("OPENAI_API_KEY")
            )
        return self._client

    async def embed(self, text: str) -> list[float]:
        """Return embedding vector for text."""
        response = await self._get_client().embeddings.create(
            model=self.model, input=text
        )
        return response.data[0].embedding
