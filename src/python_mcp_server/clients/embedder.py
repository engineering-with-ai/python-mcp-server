"""Embedders for query strings — two providers per ADR-024.

Both produce 1024d vectors so corpus dumps share schema; only the
embedded body differs.

  - Ollama Qwen3-Embedding 4b → local dev + airgapped customers
    (returns 2560d, we truncate to first 1024 via Matryoshka per ADR-024)
  - Bedrock Titan v2 (1024d) → defense + commercial cloud per ADR-024

CRITICAL parity contract: the embedder used at query time MUST match the
one used to embed the corpus at seed time (arcnode/seed). A mismatch is
silent — queries map to a different vector space than the corpus, recall
collapses to ~0 with no error to surface the bug. The cfg.yml customer
block's `llm_provider` discriminator picks runtime; the seed pipeline's
--provider flag picks the same one at build time.
"""

# Reason: PEP 563 lazy annotations — lets TYPE_CHECKING-only imports
# (boto3 stubs) work without runtime string quoting.
from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import boto3
import httpx

from ..config import BedrockSettings, ProviderSettings, OllamaSettings

if TYPE_CHECKING:
    # Type stubs only — never imported at runtime, keeps wheel slim
    from mypy_boto3_bedrock_runtime import BedrockRuntimeClient

log = logging.getLogger(__name__)

EMBEDDING_DIM = 1024  # ADR-024: same dim for cloud + airgapped; cross-DB schema


class Embedder(ABC):
    """Abstract: turn text into a vector. Provider-agnostic."""

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Return embedding vector for text. Length == EMBEDDING_DIM."""


class BedrockTitanEmbedder(Embedder):
    """Amazon Titan Text Embeddings V2 via Bedrock (native 1024d).

    Uses boto3 (sync) wrapped in asyncio.to_thread. Per-call latency is
    100-200ms; the threadpool overhead is negligible compared to a fully
    async path via aioboto3 — and avoids another transitive dep.
    """

    def __init__(self, model_id: str = "amazon.titan-embed-text-v2:0") -> None:
        self.model_id = model_id
        self._client: BedrockRuntimeClient | None = None

    def _get_client(self) -> BedrockRuntimeClient:
        if self._client is None:
            # Region inferred from AWS_REGION / default. EC2 instance role
            # provides creds in prod; local dev uses ~/.aws/credentials.
            self._client = boto3.client("bedrock-runtime")
        return self._client

    def _sync_invoke(self, text: str) -> list[float]:
        resp = self._get_client().invoke_model(
            modelId=self.model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps({"inputText": text}),
        )
        return json.loads(resp["body"].read())["embedding"]

    async def embed(self, text: str) -> list[float]:
        """Invoke Titan via boto3 in a worker thread."""
        try:
            return await asyncio.to_thread(self._sync_invoke, text)
        except Exception:
            log.exception(
                "Bedrock Titan embed failed (model=%s, input_len=%d)",
                self.model_id,
                len(text),
            )
            raise


class OllamaQwen3Embedder(Embedder):
    """Qwen3-Embedding 4b via Ollama (returns 2560d, truncated to 1024).

    Matryoshka representation learning lets us take the first 1024 dims
    and get a still-meaningful embedding at the same dim Titan emits.
    Per ADR-024 this is the airgapped + local-dev path.
    """

    def __init__(self, base_url: str, model: str = "qwen3-embedding:4b") -> None:
        # base_url should include the /v1 path so we hit the OpenAI-compat
        # endpoint (POST /v1/embeddings) — same shape as the OpenAI SDK
        # would send, easier to debug.
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def embed(self, text: str) -> list[float]:
        """POST text to Ollama's OpenAI-compat /embeddings; truncate to 1024d."""
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    f"{self.base_url}/embeddings",
                    json={"model": self.model, "input": text},
                )
                resp.raise_for_status()
                full = resp.json()["data"][0]["embedding"]
                # Matryoshka truncation — first 1024 dims of the 2560 vector.
                return full[:EMBEDDING_DIM]
        except Exception:
            log.exception(
                "Ollama Qwen3 embed failed (model=%s, base_url=%s, input_len=%d)",
                self.model,
                self.base_url,
                len(text),
            )
            raise


def make_embedder(settings: ProviderSettings) -> Embedder:
    """Construct an Embedder from a customer's resolved settings block.

    Dispatch is by type — pydantic's discriminated union has already
    validated the right field combo, so isinstance is exhaustive.
    """
    if isinstance(settings, BedrockSettings):
        return BedrockTitanEmbedder(model_id=settings.bedrock_embedding_model_id)
    if isinstance(settings, OllamaSettings):
        return OllamaQwen3Embedder(
            base_url=settings.ollama_base_url, model=settings.ollama_embedding_model
        )
    raise TypeError(f"unknown settings type: {type(settings).__name__}")
