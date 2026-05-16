"""Unit tests for the Bedrock-backed graphiti clients.

Stub boto3 invoke_model — no real AWS calls. Per the test-taxonomy
rule, this is unit-level (pure dispatch / shape mapping), not
integration: the contract being tested is "given an Anthropic-shaped
response, do we extract the right field" rather than "does Bedrock
actually answer."
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from .graphiti_bedrock import (
    BedrockCrossEncoderClient,
    BedrockEmbedderClient,
    BedrockLLMClient,
)


def _stub_invoke_response(payload: dict) -> dict:
    """Build the boto3 invoke_model response shape."""
    body = MagicMock()
    body.read.return_value = json.dumps(payload).encode()
    return {"body": body}


class TestBedrockEmbedderClient:
    """Shape-only — embedder returns the embedding field verbatim."""

    @pytest.mark.asyncio
    async def test_create_returns_embedding_from_titan_body(self) -> None:
        # Arrange
        with patch(
            "python_mcp_server.clients.graphiti_bedrock.boto3.client"
        ) as mock_client:
            mock_client.return_value.invoke_model.return_value = _stub_invoke_response(
                {"embedding": [0.1, 0.2, 0.3]}
            )

            # Act
            client = BedrockEmbedderClient()
            result = await client.create("the grid")

            # Assert
            assert result == [0.1, 0.2, 0.3]


class TestBedrockLLMClient:
    """Verify the Anthropic-on-Bedrock body shape + tool_use extraction."""

    @pytest.mark.asyncio
    async def test_generate_response_returns_text_for_plain_call(self) -> None:
        # Arrange
        from graphiti_core.prompts.models import Message

        with patch(
            "python_mcp_server.clients.graphiti_bedrock.boto3.client"
        ) as mock_client:
            mock_client.return_value.invoke_model.return_value = _stub_invoke_response(
                {
                    "content": [{"type": "text", "text": "Paris."}],
                    "stop_reason": "end_turn",
                }
            )

            # Act
            llm = BedrockLLMClient()
            result = await llm._generate_response(
                [Message(role="user", content="capital of France?")]
            )

            # Assert
            assert result == {"content": "Paris."}

    @pytest.mark.asyncio
    async def test_generate_response_extracts_tool_use_input_for_response_model(
        self,
    ) -> None:
        # Arrange
        from graphiti_core.prompts.models import Message

        class _Schema(BaseModel):
            answer: str

        with patch(
            "python_mcp_server.clients.graphiti_bedrock.boto3.client"
        ) as mock_client:
            mock_client.return_value.invoke_model.return_value = _stub_invoke_response(
                {
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "respond",
                            "input": {"answer": "Paris"},
                        }
                    ],
                    "stop_reason": "tool_use",
                }
            )

            # Act
            llm = BedrockLLMClient()
            result = await llm._generate_response(
                [Message(role="user", content="capital of France?")],
                response_model=_Schema,
            )

            # Assert
            assert result == {"answer": "Paris"}


class TestBedrockCrossEncoderClient:
    """Reranker sorts by score desc; clamps to passages length."""

    @pytest.mark.asyncio
    async def test_rank_sorts_passages_by_llm_scores(self) -> None:
        # Arrange
        llm = MagicMock(spec=BedrockLLMClient)
        llm._generate_response = AsyncMock(return_value={"scores": [10, 80, 50]})
        reranker = BedrockCrossEncoderClient(llm=llm)

        # Act
        result = await reranker.rank("query", ["low", "high", "mid"])

        # Assert — sorted desc, scores normalized to 0-1
        assert result == [("high", 0.80), ("mid", 0.50), ("low", 0.10)]

    @pytest.mark.asyncio
    async def test_rank_empty_passages_returns_empty(self) -> None:
        # Arrange
        reranker = BedrockCrossEncoderClient(llm=MagicMock(spec=BedrockLLMClient))

        # Act
        result = await reranker.rank("query", [])

        # Assert
        assert result == []
