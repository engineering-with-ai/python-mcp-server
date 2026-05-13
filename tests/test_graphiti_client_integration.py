"""Real-container integration test for GraphitiClient.from_env() Neo4j path.

Spins up a real Neo4j 5 community container and verifies that
GraphitiClient.from_env() picks the Neo4j branch, splits creds out of the
URL, actually connects, and that .search() returns a list cleanly.

OpenAI clients are stubbed at the graphiti_core import sites — this test
covers the from_env → driver → connect → search wiring, NOT Graphiti's
LLM extraction. Semantic correctness of returned facts belongs in a
higher-level e2e test with real OpenAI credentials.
"""

import os
from unittest.mock import AsyncMock, patch

import pytest
from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.embedder import EmbedderClient
from graphiti_core.llm_client import LLMClient
from testcontainers.neo4j import Neo4jContainer

from src.python_mcp_server.clients.graphiti_client import GraphitiClient

EMBED_DIM = 1024


def _graph_url_with_embedded_creds(container: Neo4jContainer) -> str:
    """Aura-style: pack user:pass into the URL for env-var packaging."""
    host = container.get_container_host_ip()
    port = container.get_exposed_port(container.port)
    return f"bolt://{container.username}:{container.password}@{host}:{port}"


@pytest.mark.asyncio
async def test_from_env_connects_to_real_neo4j_and_search_returns_list(
    neo4j: Neo4jContainer,
) -> None:
    """from_env() with GRAPH_URL → real connect → search() returns a list."""
    # Arrange — stub OpenAI defaults so search() doesn't hit the network.
    # spec=BaseClass is required so Graphiti's pydantic validation accepts them.
    mock_llm = AsyncMock(spec=LLMClient)
    mock_embedder = AsyncMock(spec=EmbedderClient)
    mock_embedder.create.return_value = [0.1] * EMBED_DIM
    mock_embedder.create_batch.return_value = [[0.1] * EMBED_DIM]
    mock_reranker = AsyncMock(spec=CrossEncoderClient)
    mock_reranker.rank.return_value = []

    os.environ["GRAPH_URL"] = _graph_url_with_embedded_creds(neo4j)

    with (
        patch("graphiti_core.graphiti.OpenAIClient", return_value=mock_llm),
        patch("graphiti_core.graphiti.OpenAIEmbedder", return_value=mock_embedder),
        patch(
            "graphiti_core.graphiti.OpenAIRerankerClient", return_value=mock_reranker
        ),
    ):
        client = GraphitiClient.from_env()

        # Smoke-test the connection: build the indices Graphiti expects.
        # This issues real Cypher against the container — proves auth + bolt work.
        await client.graphiti.build_indices_and_constraints()

        # Act
        results = await client.search("anything", limit=5)

        # Assert
        assert isinstance(
            results, list
        ), "search() should return a list even when the graph is empty"

        await client.close()
