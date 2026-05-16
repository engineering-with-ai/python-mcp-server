"""Unit tests for embedder dispatch — pure pydantic + isinstance, no infra."""

from ..config import BedrockSettings, OllamaSettings
from .embedder import (
    BedrockTitanEmbedder,
    OllamaQwen3Embedder,
    make_embedder,
)


class TestMakeEmbedder:
    """Dispatch on the discriminated-union settings type."""

    def test_bedrock_settings_yields_titan_embedder(self) -> None:
        """BedrockSettings -> BedrockTitanEmbedder with the configured model id."""
        # Arrange
        settings = BedrockSettings(
            llm_provider="bedrock",
            bedrock_chat_model_id="us.anthropic.claude-sonnet-4-6",
            bedrock_embedding_model_id="amazon.titan-embed-text-v2:0",
        )

        # Act
        embedder = make_embedder(settings)

        # Assert
        assert isinstance(embedder, BedrockTitanEmbedder)
        assert embedder.model_id == "amazon.titan-embed-text-v2:0"

    def test_ollama_settings_yields_qwen3_embedder(self) -> None:
        """OllamaSettings -> OllamaQwen3Embedder pointed at the configured URL."""
        # Arrange
        settings = OllamaSettings(
            llm_provider="ollama",
            ollama_base_url="http://173.211.12.43:11434/v1",
            ollama_chat_model="qwen3.6:35b",
            ollama_embedding_model="qwen3-embedding:4b",
        )

        # Act
        embedder = make_embedder(settings)

        # Assert
        assert isinstance(embedder, OllamaQwen3Embedder)
        assert embedder.model == "qwen3-embedding:4b"
        assert embedder.base_url == "http://173.211.12.43:11434/v1"
