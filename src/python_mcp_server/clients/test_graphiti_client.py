"""Unit tests for GraphitiClient.from_env() factory."""

from unittest.mock import patch

import pytest

from .graphiti_client import GraphitiClient


class TestFromEnvFactory:
    """Branch selection: GRAPH_URL → Neo4j, NEPTUNE_HOST+AOSS_HOST → Neptune."""

    def test_graph_url_env_constructs_neo4j_backed_client(self) -> None:
        """GRAPH_URL → Graphiti built with split (uri, user, password)."""
        # Arrange — embedded creds get pulled out of the URL
        env = {"GRAPH_URL": "neo4j+s://alice:s3cret@host.example:7687"}
        with (
            patch.dict("os.environ", env, clear=True),
            patch(
                "python_mcp_server.clients.graphiti_client.Graphiti"
            ) as mock_graphiti_class,
        ):
            # Act
            client = GraphitiClient.from_env()

            # Assert
            mock_graphiti_class.assert_called_once_with(
                uri="neo4j+s://host.example:7687",
                user="alice",
                password="s3cret",  # noqa: S106  # nosec B106
            )
            assert client.graphiti is mock_graphiti_class.return_value

    def test_graph_url_without_creds_passes_none_for_user_password(self) -> None:
        """No creds in URL → user/password are None (suits NEO4J_AUTH=none deploys)."""
        # Arrange
        env = {"GRAPH_URL": "bolt://host.example:7687"}
        with (
            patch.dict("os.environ", env, clear=True),
            patch(
                "python_mcp_server.clients.graphiti_client.Graphiti"
            ) as mock_graphiti_class,
        ):
            # Act
            GraphitiClient.from_env()

            # Assert
            mock_graphiti_class.assert_called_once_with(
                uri="bolt://host.example:7687", user=None, password=None
            )

    def test_url_encoded_creds_are_unquoted(self) -> None:
        """%-encoded creds in GRAPH_URL get decoded before reaching Neo4j.

        Reason: shared dev passwords often contain @ and ! which must be
        encoded for asyncpg's URL parser; the encoded form must also work
        for the graph path so one URL works for both consumers.
        """
        # Arrange — password 'REDACTED' encoded as %21 and %401
        env = {
            "GRAPH_URL": "neo4j://neo4j:REDACTED@host.example:7687"
        }
        with (
            patch.dict("os.environ", env, clear=True),
            patch(
                "python_mcp_server.clients.graphiti_client.Graphiti"
            ) as mock_graphiti_class,
        ):
            # Act
            GraphitiClient.from_env()

            # Assert — Neo4j gets the *decoded* password, not the encoded string
            mock_graphiti_class.assert_called_once_with(
                uri="neo4j://host.example:7687",
                user="neo4j",
                password="REDACTED",  # noqa: S106  # nosec B106
            )

    def test_neptune_env_constructs_neptune_backed_client(self) -> None:
        """NEPTUNE_HOST + AOSS_HOST → Graphiti with NeptuneDriver + Bedrock clients.

        Per #64: cloud customers must NOT reach for OPENAI_API_KEY at runtime.
        The Neptune branch wires graphiti's embedder/llm/cross_encoder to
        Bedrock-backed implementations from graphiti_bedrock.
        """
        # Arrange
        env = {
            "NEPTUNE_HOST": "my-cluster.amazonaws.com",
            "AOSS_HOST": "my-aoss.amazonaws.com",
        }
        with (
            patch.dict("os.environ", env, clear=True),
            patch(
                "python_mcp_server.clients.graphiti_client.Graphiti"
            ) as mock_graphiti_class,
            patch(
                "python_mcp_server.clients.graphiti_client.NeptuneDriver"
            ) as mock_driver_class,
            patch(
                "python_mcp_server.clients.graphiti_bedrock.boto3.client"
            ),  # graphiti_bedrock constructs boto3 clients on init; stub
        ):
            # Act
            client = GraphitiClient.from_env()

            # Assert
            mock_driver_class.assert_called_once_with(
                host="neptune-db://my-cluster.amazonaws.com",
                aoss_host="my-aoss.amazonaws.com",
            )
            mock_graphiti_class.assert_called_once()
            kwargs = mock_graphiti_class.call_args.kwargs
            assert kwargs["graph_driver"] is mock_driver_class.return_value
            # The Bedrock clients are wired so graphiti doesn't reach for OpenAI
            from .graphiti_bedrock import (
                BedrockCrossEncoderClient,
                BedrockEmbedderClient,
                BedrockLLMClient,
            )

            assert isinstance(kwargs["embedder"], BedrockEmbedderClient)
            assert isinstance(kwargs["llm_client"], BedrockLLMClient)
            assert isinstance(kwargs["cross_encoder"], BedrockCrossEncoderClient)
            assert client.graphiti is mock_graphiti_class.return_value

    def test_missing_env_raises_runtime_error(self) -> None:
        """No GRAPH_URL and no NEPTUNE_HOST/AOSS_HOST → fail loudly, no defaults."""
        # Arrange + Act + Assert
        with (
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(RuntimeError, match="no graph backend configured"),
        ):
            GraphitiClient.from_env()
