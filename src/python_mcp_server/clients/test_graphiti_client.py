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

    def test_neptune_env_constructs_neptune_backed_client(self) -> None:
        """NEPTUNE_HOST + AOSS_HOST → Graphiti built with a NeptuneDriver."""
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
        ):
            # Act
            client = GraphitiClient.from_env()

            # Assert
            mock_driver_class.assert_called_once_with(
                host="neptune-db://my-cluster.amazonaws.com",
                aoss_host="my-aoss.amazonaws.com",
            )
            mock_graphiti_class.assert_called_once_with(
                graph_driver=mock_driver_class.return_value
            )
            assert client.graphiti is mock_graphiti_class.return_value

    def test_missing_env_raises_runtime_error(self) -> None:
        """No GRAPH_URL and no NEPTUNE_HOST/AOSS_HOST → fail loudly, no defaults."""
        # Arrange + Act + Assert
        with (
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(RuntimeError, match="no graph backend configured"),
        ):
            GraphitiClient.from_env()
