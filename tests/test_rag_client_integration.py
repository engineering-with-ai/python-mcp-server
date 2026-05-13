"""Real-container integration test for RAGClient hybrid search.

Spins up a pgvector container, seeds three known rows, and verifies that
RAGClient.search() exercises both the cosine and BM25 legs and fuses
them via RRF. The Embedder is stubbed so the cosine query vector is
deterministic — the test owns both sides of the similarity.
"""

from unittest.mock import AsyncMock

import asyncpg
import pytest
from pgvector.asyncpg import register_vector
from testcontainers.postgres import PostgresContainer

from src.python_mcp_server.clients.embedder import Embedder
from src.python_mcp_server.clients.rag_client import RAGClient
from urllib.parse import quote_plus

# Reason: 4 dims keeps the test readable; mocked embedder returns matching shape
EMBED_DIM = 4
QUERY_VECTOR = [1.0, 0.0, 0.0, 0.0]


async def _seed_energy_embeddings(
    *, host: str, port: int, user: str, password: str, database: str
) -> None:
    """Create the energy_embeddings schema rag_client.search() expects.

    Mirrors production columns: id, title, content, book, section_level,
    analysis_relevance, effective_from/until, normative, embedding (vector),
    and content_tsv (tsvector for BM25).

    Uses keyword-arg connect to dodge URL-encoding the testcontainers password.
    """
    conn = await asyncpg.connect(
        host=host, port=port, user=user, password=password, database=database
    )
    try:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await register_vector(conn)
        await conn.execute(f"""
            CREATE TABLE energy_embeddings (
                id TEXT PRIMARY KEY,
                title TEXT,
                content TEXT,
                book TEXT,
                section_level TEXT,
                analysis_relevance TEXT,
                effective_from TIMESTAMPTZ,
                effective_until TIMESTAMPTZ,
                normative BOOLEAN,
                embedding vector({EMBED_DIM}),
                content_tsv tsvector
            )
            """)
        rows = [
            # docA: vector matches QUERY_VECTOR exactly, content unrelated
            (
                "docA",
                "alpha doc",
                "generic content nothing special",
                [1.0, 0.0, 0.0, 0.0],
            ),
            # docB: irrelevant on both legs
            ("docB", "beta doc", "completely unrelated material", [0.0, 1.0, 0.0, 0.0]),
            # docC: vector far from query, but contains the unique BM25 term
            (
                "docC",
                "gamma doc",
                "transformers and reactive power compensation",
                [0.0, 0.0, 1.0, 0.0],
            ),
        ]
        for doc_id, title, content, vec in rows:
            await conn.execute(
                """
                INSERT INTO energy_embeddings
                  (id, title, content, embedding, content_tsv)
                VALUES ($1, $2, $3, $4, to_tsvector('english', $3))
                """,
                doc_id,
                title,
                content,
                vec,
            )
    finally:
        await conn.close()


def _db_url_from_container(container: PostgresContainer) -> str:
    """Build a postgres URL from a testcontainer, quote_plus'ing the password."""
    return (
        f"postgresql://{container.username}:{quote_plus(container.password)}"
        f"@{container.get_container_host_ip()}"
        f":{int(container.get_exposed_port(5432))}/{container.dbname}"
    )


@pytest.mark.asyncio
async def test_search_fuses_cosine_and_bm25_against_real_pgvector(
    postgres_pgvector: PostgresContainer,
) -> None:
    """search() against a real pgvector DB returns the cosine *and* BM25 winners."""
    # Arrange
    db_url = _db_url_from_container(postgres_pgvector)
    await _seed_energy_embeddings(
        host=postgres_pgvector.get_container_host_ip(),
        port=int(postgres_pgvector.get_exposed_port(5432)),
        user=postgres_pgvector.username,
        password=postgres_pgvector.password,
        database=postgres_pgvector.dbname,
    )

    embedder = AsyncMock(spec=Embedder)
    embedder.embed.return_value = QUERY_VECTOR

    client = RAGClient(db_url=db_url, embedder=embedder, table_name="energy_embeddings")

    # Act: BM25 leg keys on "transformers" (only in docC); cosine leg
    # returns docA via stubbed query vector.
    results = await client.search("transformers", limit=10)

    # Assert
    result_ids = {r.id for r in results}
    assert "docA" in result_ids, "cosine winner missing — vector leg didn't fire"
    assert "docC" in result_ids, "BM25 winner missing — text leg didn't fire"
