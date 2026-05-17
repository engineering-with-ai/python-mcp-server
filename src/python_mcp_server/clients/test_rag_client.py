"""Unit tests for rag_client — RRF fusion logic + hybrid search wiring."""

from unittest.mock import AsyncMock, patch

import pytest

from .embedder import Embedder
from .rag_client import RAGClient, _connect, rrf_fuse


@pytest.mark.asyncio
async def test_connect_unquotes_password_and_uses_kwargs() -> None:
    """_connect() parses an encoded URL and passes kwargs (no raw URL).

    Reason: asyncpg's URL parser splits on the FIRST `@` so a password
    containing `@` (common in shared dev creds) breaks DNS resolution.
    Going through kwargs avoids that parser; unquote() returns the real
    password to asyncpg.
    """
    # Arrange — 'REDACTED' encoded ends in %40%21
    url = "postgres://postgres:REDACTED@host.example:5432/postgres"
    with patch(
        "src.python_mcp_server.clients.rag_client.asyncpg.connect"
    ) as mock_connect:
        mock_connect.return_value = AsyncMock()

        # Act
        await _connect(url)

        # Assert — asyncpg gets decoded password via kwargs, no URL
        mock_connect.assert_called_once_with(
            host="host.example",
            port=5432,
            user="postgres",
            password="REDACTED",  # noqa: S106  # nosec B106
            database="postgres",
        )


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
    # Arrange — cosine returns [doc1, doc2]; bm25 returns [doc1, doc3]
    cosine_rows = [_row("doc1"), _row("doc2")]
    bm25_rows = [_row("doc1"), _row("doc3")]
    with patch(
        "src.python_mcp_server.clients.rag_client.asyncpg.connect"
    ) as mock_connect:
        mock_conn = AsyncMock()
        mock_connect.return_value = mock_conn
        mock_conn.fetch.side_effect = [cosine_rows, bm25_rows]

        embedder = AsyncMock(spec=Embedder)
        embedder.embed.return_value = [0.1] * 1536
        client = RAGClient(
            db_url="postgresql://test:test_pw@localhost:5432/test",
            embedder=embedder,
            table_name="energy_embeddings",
        )

        # Act
        results = await client.search("test query")

        # Assert: doc1 appears in both rankings → RRF ranks it first
        assert len(results) == 3
        assert results[0].id == "doc1"
        assert {r.id for r in results} == {"doc1", "doc2", "doc3"}
