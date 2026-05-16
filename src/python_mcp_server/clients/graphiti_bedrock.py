"""Bedrock-backed graphiti clients for cloud (commercial + defense) customers.

graphiti-core only ships OpenAI / Anthropic-direct / Gemini / Groq / Azure
clients out of the box — no Bedrock. Cloud customers run via Bedrock per
ADR-024 (no third-party API keys), so we subclass graphiti's three
client base classes here.

Three components:
  - BedrockEmbedderClient    → Titan v2 (1024d native)
  - BedrockLLMClient         → Sonnet 4.6 via CRIS (us.* inference profile)
  - BedrockCrossEncoderClient → LLM-as-reranker over Sonnet 4.6

Anthropic structured output uses the native `tool_use` API (passing
`response_model.model_json_schema()` as a forced tool) — far more
reliable than prompt-engineered JSON.
"""

from __future__ import annotations

import asyncio
import json
import logging
import typing
from collections.abc import Iterable
from typing import TYPE_CHECKING

import boto3
from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.embedder.client import EmbedderClient, EmbedderConfig
from graphiti_core.llm_client.client import LLMClient
from graphiti_core.llm_client.config import DEFAULT_MAX_TOKENS, LLMConfig, ModelSize
from graphiti_core.prompts.models import Message
from pydantic import BaseModel

if TYPE_CHECKING:
    from mypy_boto3_bedrock_runtime import BedrockRuntimeClient

log = logging.getLogger(__name__)

TITAN_MODEL_ID = "amazon.titan-embed-text-v2:0"
SONNET_INFERENCE_PROFILE = "us.anthropic.claude-sonnet-4-6"
ANTHROPIC_VERSION = "bedrock-2023-05-31"


class BedrockEmbedderClient(EmbedderClient):
    """Titan Embed v2 wrapped for graphiti. 1024d native — matches ADR-024."""

    def __init__(self, model_id: str = TITAN_MODEL_ID) -> None:
        self.config = EmbedderConfig(embedding_dim=1024)
        self.model_id = model_id
        self._client: BedrockRuntimeClient = boto3.client("bedrock-runtime")

    def _invoke_sync(self, text: str) -> list[float]:
        resp = self._client.invoke_model(
            modelId=self.model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps({"inputText": text}),
        )
        return json.loads(resp["body"].read())["embedding"]

    async def create(
        self,
        input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]],
    ) -> list[float]:
        """Graphiti always passes str or list[str] in practice."""
        if isinstance(input_data, str):
            return await asyncio.to_thread(self._invoke_sync, input_data)
        if (
            isinstance(input_data, list)
            and input_data
            and isinstance(input_data[0], str)
        ):
            # Single combined embedding for list-of-string input — matches
            # OpenAI's behavior when graphiti passes a batch on this code path.
            strs: list[str] = [s for s in input_data if isinstance(s, str)]
            return await asyncio.to_thread(self._invoke_sync, " ".join(strs))
        raise TypeError(f"unsupported input_data type: {type(input_data).__name__}")

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        """Per-item invoke — Titan has no batch endpoint; loop is cheapest."""
        return await asyncio.gather(
            *(asyncio.to_thread(self._invoke_sync, t) for t in input_data_list)
        )


class BedrockLLMClient(LLMClient):
    """Sonnet 4.6 via CRIS. Uses Anthropic tool_use API for structured output."""

    def __init__(self, model_id: str = SONNET_INFERENCE_PROFILE) -> None:
        super().__init__(LLMConfig(model=model_id))
        self.model_id = model_id
        self._client: BedrockRuntimeClient = boto3.client("bedrock-runtime")

    def _split_system(self, messages: list[Message]) -> tuple[str, list[dict]]:
        """Anthropic API takes system prompt as a separate field, not in messages."""
        system_parts: list[str] = []
        anthropic_msgs: list[dict] = []
        for m in messages:
            if m.role == "system":
                system_parts.append(m.content)
            else:
                anthropic_msgs.append({"role": m.role, "content": m.content})
        return "\n\n".join(system_parts), anthropic_msgs

    def _invoke_sync(self, body: dict) -> dict:
        resp = self._client.invoke_model(
            modelId=self.model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body),
        )
        return json.loads(resp["body"].read())

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model_size: ModelSize = ModelSize.medium,  # noqa: ARG002 — Bedrock has no size knob
    ) -> dict[str, typing.Any]:
        """Returns dict; if response_model given, parsed JSON matching that schema."""
        system, anthropic_msgs = self._split_system(messages)
        body: dict = {
            "anthropic_version": ANTHROPIC_VERSION,
            "max_tokens": max_tokens,
            "messages": anthropic_msgs,
        }
        if system:
            body["system"] = system

        if response_model is not None:
            # Use Anthropic's tool_use API for structured output —
            # native schema enforcement vs prompt-engineered JSON.
            body["tools"] = [
                {
                    "name": "respond",
                    "description": "Respond with the structured data.",
                    "input_schema": response_model.model_json_schema(),
                }
            ]
            body["tool_choice"] = {"type": "tool", "name": "respond"}

        result = await asyncio.to_thread(self._invoke_sync, body)

        if response_model is not None:
            for block in result.get("content", []):
                if block.get("type") == "tool_use":
                    return block["input"]
            raise RuntimeError(
                f"Bedrock didn't return a tool_use block: {result.get('stop_reason')}"
            )

        # Plain text response
        text = "".join(
            b.get("text", "")
            for b in result.get("content", [])
            if b.get("type") == "text"
        )
        return {"content": text}


class BedrockCrossEncoderClient(CrossEncoderClient):
    """LLM-as-reranker — Sonnet scores each passage's relevance to the query."""

    def __init__(self, llm: BedrockLLMClient | None = None) -> None:
        self._llm = llm or BedrockLLMClient()

    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        """Score each passage 0-1 by relevance. Returns sorted desc."""
        if not passages:
            return []
        # One LLM call ranks all passages — much cheaper than per-passage calls.
        numbered = "\n".join(f"[{i}] {p[:500]}" for i, p in enumerate(passages))
        prompt = (
            f"Query: {query}\n\n"
            f"Passages:\n{numbered}\n\n"
            "Score each passage's relevance to the query on a scale of 0-100."
        )
        messages = [Message(role="user", content=prompt)]

        class _Scores(BaseModel):
            scores: list[int]

        try:
            result = await self._llm._generate_response(
                messages, response_model=_Scores
            )
            scores = result["scores"]
            if len(scores) != len(passages):
                # Pad/truncate to match passages length
                scores = (scores + [0] * len(passages))[: len(passages)]
        except Exception:
            log.exception("reranker failed; falling back to identity ranking")
            scores = [50] * len(passages)

        pairs = [(p, s / 100.0) for p, s in zip(passages, scores, strict=True)]
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs
