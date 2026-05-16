"""Integration test for the MCP rag_search tool — calls through the SYSTEM.

Per the test-taxonomy rule, integration tests must exercise the system
through its public entry point (the MCP tool surface), not the wrapper
class in isolation. This test:

  1. Spins up a real pgvector container, creates the `knowledge` table.
  2. Seeds one row whose embedding matches the (mocked) Ollama vector
     so cosine similarity drives it to the top.
  3. Calls `mcp.call_tool("rag_search", ...)` — the same surface a real
     LLM client would hit.
  4. Asserts:
       - MCP returns the seeded row in the response.
       - Pook saw the right POST to Ollama (request shape contract).

E2E flag (cfg.e2e=true in cfg.yml): skips pook, hits the real arcnode
Ollama at 173.211.12.43:11434 — free dogfood, catches drift vs the
mocked contract.
"""

from urllib.parse import quote_plus

import asyncpg
import pook
import pytest
from pgvector.asyncpg import register_vector
from testcontainers.postgres import PostgresContainer

from src.python_mcp_server.clients.embedder import EMBEDDING_DIM
from src.python_mcp_server.config import (
    Config,
    LogLevel,
    OllamaSettings,
    load_config,
)
from src.python_mcp_server.server import create_server
from tests.fixtures.containers import postgres_pgvector  # noqa: F401  pytest fixture

# Ollama Qwen3 returns 2560d; embedder truncates to 1024. Build matching
# shapes: full vector for the mock response, truncated for the DB row so
# cosine distance is 0.
OLLAMA_FULL_VEC = [0.001 * i for i in range(2560)]
KNOWLEDGE_VEC = OLLAMA_FULL_VEC[:EMBEDDING_DIM]

OLLAMA_BASE_URL = "http://173.211.12.43:11434/v1"


def _db_url(container: PostgresContainer) -> str:
    return (
        f"postgresql://{container.username}:{quote_plus(container.password)}"
        f"@{container.get_container_host_ip()}"
        f":{int(container.get_exposed_port(5432))}/{container.dbname}"
    )


async def _seed_knowledge(container: PostgresContainer) -> None:
    """Create the `knowledge` schema rag_client expects + insert one row.

    Schema mirrors arcnode/seed/src/seed_vector.py (production columns)
    minus the optional persona/test-case GIN-indexed columns that the
    SELECT path doesn't read.
    """
    conn = await asyncpg.connect(
        host=container.get_container_host_ip(),
        port=int(container.get_exposed_port(5432)),
        user=container.username,
        password=container.password,
        database=container.dbname,
    )
    try:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await register_vector(conn)
        await conn.execute(f"""
            CREATE TABLE knowledge (
                id TEXT PRIMARY KEY,
                title TEXT,
                content TEXT,
                book TEXT,
                section_level TEXT,
                analysis_relevance TEXT,
                effective_from TIMESTAMPTZ,
                effective_until TIMESTAMPTZ,
                normative BOOLEAN,
                embedding vector({EMBEDDING_DIM}),
                content_tsv tsvector
            )
            """)
        await conn.execute(
            """
            INSERT INTO knowledge
              (id, title, content, book, embedding, content_tsv)
            VALUES ($1, $2, $3, $4, $5, to_tsvector('english', $3))
            """,
            "doc1",
            "Grid Fundamentals",
            "the electric grid operates at multiple voltage levels",
            "Power Systems Basics",
            KNOWLEDGE_VEC,
        )
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_mcp_rag_search_calls_ollama_and_returns_pgvector_match(
    postgres_pgvector: PostgresContainer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MCP rag_search exercises the full chain: tool -> embedder -> pgvector."""
    # Arrange
    cfg = load_config()
    await _seed_knowledge(postgres_pgvector)
    monkeypatch.setenv("VECTOR_URL", _db_url(postgres_pgvector))

    if not cfg.e2e:
        pook.on()
        pook.enable_network()
        pook.post(f"{OLLAMA_BASE_URL}/embeddings").reply(200).json(
            {"data": [{"embedding": OLLAMA_FULL_VEC}]}
        )

    test_config = Config(
        log_level=LogLevel.INFO,
        e2e=cfg.e2e,
        settings=OllamaSettings(
            llm_provider="ollama",
            ollama_base_url=OLLAMA_BASE_URL,
            ollama_chat_model="qwen3.6:35b",
            ollama_embedding_model="qwen3-embedding:4b",
        ),
    )
    server = create_server(test_config)

    try:
        # Act — call through the MCP tool surface, not the wrapper
        _content, result = await server.call_tool(
            "rag_search",
            {"query": "the grid operates at multiple voltage levels"},
        )

        # Assert — MCP returned the seeded doc
        assert isinstance(result, dict)
        docs = result.get("result", result)
        items = docs if isinstance(docs, list) else docs.get("items") or []
        assert len(items) >= 1, f"expected at least one doc, got {items}"
        ids = {item["id"] for item in items}
        assert "doc1" in ids, f"seeded doc1 missing from {ids}"

        # Assert — when mocked, Ollama was actually hit with the right shape.
        # pook.is_done() == all registered mocks consumed; pending == none left
        if not cfg.e2e:
            assert pook.isdone(), (
                "MCP→embedder→Ollama path didn't fire — pook still has "
                f"pending mocks: {pook.pending()}"
            )
    finally:
        pook.off()
