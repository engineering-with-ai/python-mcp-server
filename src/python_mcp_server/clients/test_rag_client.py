"""Unit tests for rag_client — RRF fusion logic + hybrid search wiring."""

from unittest.mock import AsyncMock, patch

import pytest

from ..config import PostgresConfig
from .embedder import Embedder
from .rag_client import RAGClient, rrf_fuse


def test_rrf_fuse_ranks_doc_in_both_rankings_first() -> None:
    """A doc appearing in both rankings outranks docs in only one."""
    # Arrange
    cosine_ranking = ["doc1", "doc2"]
    bm25_ranking = ["doc1", "doc3"]

    # Act
    fused = rrf_fuse([cosine_ranking, bm25_ranking], k=60)

    # Assert
    assert fused[0][0] == "doc1"


def _row(doc_id: str, content: str = "") -> dict[str, object]:
    """Build a minimal energy_embeddings row for mock fetch."""
    return {
        "id": doc_id,
        "title": "",
        "content": content or doc_id,
        "book": "",
        "section_level": "",
        "analysis_relevance": "",
    }


@pytest.mark.asyncio
async def test_rag_client_search_fuses_cosine_and_bm25_rankings() -> None:
    """search() issues both cosine and BM25 queries and fuses via RRF."""
    # Arrange
    test_config = PostgresConfig(
        host="localhost",
        port=5432,
        database="test",
        user="test",
        embeddings_table="energy_embeddings",
        embedding_model="text-embedding-3-small",
    )
    # cosine returns [doc1, doc2]; bm25 returns [doc1, doc3]
    cosine_rows = [_row("doc1"), _row("doc2")]
    bm25_rows = [_row("doc1"), _row("doc3")]
    with (
        patch(
            "src.python_mcp_server.clients.rag_client.asyncpg.connect"
        ) as mock_connect,
        patch("src.python_mcp_server.clients.rag_client.os.getenv") as mock_getenv,
    ):
        mock_getenv.return_value = "test_pw"
        mock_conn = AsyncMock()
        mock_connect.return_value = mock_conn
        mock_conn.fetch.side_effect = [cosine_rows, bm25_rows]

        embedder = AsyncMock(spec=Embedder)
        embedder.embed.return_value = [0.1] * 1536
        client = RAGClient(test_config, embedder=embedder)

        # Act
        results = await client.search("test query")

        # Assert: doc1 appears in both rankings → RRF ranks it first
        assert len(results) == 3
        assert results[0].id == "doc1"
        assert {r.id for r in results} == {"doc1", "doc2", "doc3"}
