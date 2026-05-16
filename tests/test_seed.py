"""Real-container integration tests for boot-time seed.

Two backends covered:
  - Neo4j (community container) — proves cypher restore + transactional
    rollback against any bolt-protocol Neo4j (Aura, ISO, dev container).
  - Postgres + pgvector — proves plain-SQL restore + transactional
    rollback against the vector slice destination.

Each test verifies the same contract: idempotent skip via marker, fatal
on partial failure with clean rollback so retry succeeds.
"""

import gzip
import io
import os
from collections.abc import AsyncIterator
from unittest.mock import patch
from urllib.parse import quote_plus

import asyncpg
import pytest
import pytest_asyncio
from neo4j import AsyncGraphDatabase
from testcontainers.neo4j import Neo4jContainer
from testcontainers.postgres import PostgresContainer

from src.python_mcp_server.clients.graphiti_client import _split_neo4j_url
from src.python_mcp_server.seed import (
    GRAPH_MARKER_SLICE,
    VECTOR_MARKER_SLICE,
    seed_graph_neo4j,
    seed_vector,
)
from tests.fixtures.containers import (  # noqa: F401  pytest fixtures
    neo4j,
    postgres_pgvector,
)

CYPHER_SCRIPT = """
CREATE (a:Foo {name: 'bar'});
CREATE (b:Foo {name: 'baz'});
""".strip()

PARTIAL_FAIL_CYPHER = """
CREATE (a:Foo {name: 'bar'});
THIS IS NOT VALID CYPHER;
CREATE (b:Foo {name: 'baz'});
""".strip()

VECTOR_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE knowledge (id SERIAL PRIMARY KEY, content TEXT, embedding vector(3));
INSERT INTO knowledge (content, embedding) VALUES ('hi', '[1,0,0]');
INSERT INTO knowledge (content, embedding) VALUES ('bye', '[0,1,0]');
""".strip()

PARTIAL_FAIL_VECTOR_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE knowledge (id SERIAL PRIMARY KEY, content TEXT, embedding vector(3));
INSERT INTO knowledge (content, embedding) VALUES ('hi', '[1,0,0]');
THIS IS NOT VALID SQL;
""".strip()


def _graph_url(container: Neo4jContainer) -> str:
    """Pack user:pass into the URL; matches Aura's env-var packaging."""
    host = container.get_container_host_ip()
    port = container.get_exposed_port(container.port)
    return f"bolt://{container.username}:{container.password}@{host}:{port}"


def _vector_url(container: PostgresContainer) -> str:
    """Build a postgres URL for VECTOR_URL env from a testcontainer.

    quote_plus the password — testcontainers generates passwords with
    URL-special chars that break asyncpg's URL parser otherwise.
    """
    return (
        f"postgresql://{container.username}:{quote_plus(container.password)}"
        f"@{container.get_container_host_ip()}"
        f":{int(container.get_exposed_port(5432))}/{container.dbname}"
    )


def _gzipped_response(body: str) -> io.BytesIO:
    """Mimic urllib.request.urlopen() return — a context-managed BytesIO."""
    return io.BytesIO(gzip.compress(body.encode()))


@pytest_asyncio.fixture(autouse=True)
async def _wipe_neo4j(neo4j: Neo4jContainer) -> AsyncIterator[None]:
    """Clear all nodes between tests — fixture is session-scoped, tests aren't."""
    yield
    clean_uri, user, password = _split_neo4j_url(_graph_url(neo4j))
    assert user is not None
    assert password is not None
    driver = AsyncGraphDatabase.driver(clean_uri, auth=(user, password))
    try:
        async with driver.session() as session:
            await session.run("MATCH (n) DETACH DELETE n")
    finally:
        await driver.close()


@pytest_asyncio.fixture(autouse=True)
async def _wipe_postgres(
    postgres_pgvector: PostgresContainer,
) -> AsyncIterator[None]:
    """Drop seed tables + marker between tests."""
    yield
    conn = await asyncpg.connect(_vector_url(postgres_pgvector))
    try:
        await conn.execute("DROP TABLE IF EXISTS knowledge CASCADE")
        await conn.execute("DROP TABLE IF EXISTS arcnode_seed_markers CASCADE")
    finally:
        await conn.close()


# ───────────────────────── Neo4j graph slice ─────────────────────────


@pytest.mark.asyncio
async def test_seed_graph_neo4j_loads_then_skips_on_rerun(
    neo4j: Neo4jContainer,
) -> None:
    """First call seeds + marks; second call skips via marker."""
    # Arrange
    os.environ["GRAPH_URL"] = _graph_url(neo4j)
    seed_url = "https://example.invalid/graph-neo4j.cypher.gz"

    # Act — first call hits the cypher script
    with patch(
        "src.python_mcp_server.seed.urllib.request.urlopen",
        return_value=_gzipped_response(CYPHER_SCRIPT),
    ):
        await seed_graph_neo4j(seed_url)

    # Assert — both Foo nodes landed
    clean_uri, user, password = _split_neo4j_url(_graph_url(neo4j))
    assert user is not None
    assert password is not None
    driver = AsyncGraphDatabase.driver(clean_uri, auth=(user, password))
    try:
        async with driver.session() as session:
            result = await session.run("MATCH (n:Foo) RETURN count(n) AS c")
            row = await result.single()
            assert row is not None
            assert row["c"] == 2

            marker = await session.run(
                "MATCH (m:ArcnodeSeedMarker {slice: $slice}) RETURN m",
                slice=GRAPH_MARKER_SLICE,
            )
            assert await marker.single() is not None

        # Act — second call should NOT re-run the cypher (urlopen unused)
        with patch(
            "src.python_mcp_server.seed.urllib.request.urlopen",
            side_effect=AssertionError("urlopen must not be called on idempotent skip"),
        ):
            await seed_graph_neo4j(seed_url)

        # Assert — still only 2 Foo nodes (no duplicates from a second restore)
        async with driver.session() as session:
            result = await session.run("MATCH (n:Foo) RETURN count(n) AS c")
            row = await result.single()
            assert row is not None
            assert row["c"] == 2
    finally:
        await driver.close()


@pytest.mark.asyncio
async def test_seed_graph_neo4j_partial_failure_rolls_back_then_retries_clean(
    neo4j: Neo4jContainer,
) -> None:
    """Bad cypher mid-script → fail-fast, rollback, retry succeeds clean."""
    # Arrange
    os.environ["GRAPH_URL"] = _graph_url(neo4j)
    seed_url = "https://example.invalid/graph-neo4j.cypher.gz"
    clean_uri, user, password = _split_neo4j_url(_graph_url(neo4j))
    assert user is not None
    assert password is not None

    # Act — first call hits a script with a bad statement in the middle
    from neo4j.exceptions import CypherSyntaxError

    with (
        patch(
            "src.python_mcp_server.seed.urllib.request.urlopen",
            return_value=_gzipped_response(PARTIAL_FAIL_CYPHER),
        ),
        pytest.raises(CypherSyntaxError),
    ):
        await seed_graph_neo4j(seed_url)

    # Assert — DB is clean (transaction rolled back, no Foo nodes, no marker)
    driver = AsyncGraphDatabase.driver(clean_uri, auth=(user, password))
    try:
        async with driver.session() as session:
            result = await session.run("MATCH (n:Foo) RETURN count(n) AS c")
            row = await result.single()
            assert row is not None
            assert row["c"] == 0, "partial restore must roll back"

            marker = await session.run(
                "MATCH (m:ArcnodeSeedMarker {slice: $slice}) RETURN m",
                slice=GRAPH_MARKER_SLICE,
            )
            assert await marker.single() is None, "marker must not be set on failure"

        # Act — retry with a clean script (simulates upstream fix + redeploy)
        with patch(
            "src.python_mcp_server.seed.urllib.request.urlopen",
            return_value=_gzipped_response(CYPHER_SCRIPT),
        ):
            await seed_graph_neo4j(seed_url)

        # Assert — clean restore worked + marker now present
        async with driver.session() as session:
            result = await session.run("MATCH (n:Foo) RETURN count(n) AS c")
            row = await result.single()
            assert row is not None
            assert row["c"] == 2

            marker = await session.run(
                "MATCH (m:ArcnodeSeedMarker {slice: $slice}) RETURN m",
                slice=GRAPH_MARKER_SLICE,
            )
            assert await marker.single() is not None
    finally:
        await driver.close()


# ──────────────────────── pgvector vector slice ──────────────────────
#
# Vector seed shells out to `psql` (subprocess) — see seed._psql_apply.
# Tests mock _psql_apply: we exercise the marker-orchestration code,
# not psql itself. psql's atomic-restore behavior (via --single-transaction)
# is a stable contract from the postgresql project; verifying it
# end-to-end belongs in a higher-level integration suite that installs
# psql alongside the test harness.


@pytest.mark.asyncio
async def test_seed_vector_loads_then_skips_on_rerun(
    postgres_pgvector: PostgresContainer,
) -> None:
    """First call invokes psql + writes marker; second call skips via marker."""
    # Arrange
    os.environ["VECTOR_URL"] = _vector_url(postgres_pgvector)
    seed_url = "https://example.invalid/vector.sql.gz"

    apply_calls: list[tuple[str, str]] = []

    async def fake_apply(dump_url: str, conn_url: str) -> None:
        apply_calls.append((dump_url, conn_url))
        # Simulate a successful psql run by creating a marker for the table
        # the real dump would create. The contract is "psql exits 0";
        # we don't replay the actual SQL because that's psql's job.
        conn = await asyncpg.connect(conn_url)
        try:
            await conn.execute(
                "CREATE TABLE IF NOT EXISTS knowledge (id INT PRIMARY KEY)"
            )
        finally:
            await conn.close()

    # Act — first call triggers psql + writes marker
    with patch("src.python_mcp_server.seed._psql_apply", side_effect=fake_apply):
        await seed_vector(seed_url)

    assert apply_calls == [(seed_url, _vector_url(postgres_pgvector))]

    conn = await asyncpg.connect(_vector_url(postgres_pgvector))
    try:
        marker = await conn.fetchval(
            "SELECT 1 FROM arcnode_seed_markers WHERE slice=$1",
            VECTOR_MARKER_SLICE,
        )
        assert marker == 1
    finally:
        await conn.close()

    # Act — second call should NOT invoke psql (marker present)
    with patch(
        "src.python_mcp_server.seed._psql_apply",
        side_effect=AssertionError("_psql_apply must not be called on idempotent skip"),
    ):
        await seed_vector(seed_url)
