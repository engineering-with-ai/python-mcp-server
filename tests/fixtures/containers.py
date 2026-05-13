"""Session-scoped testcontainer fixtures for real-DB integration tests.

Skips cleanly when docker isn't reachable so unit suites stay green on
machines without docker.
"""

from collections.abc import Iterator

import pytest
from testcontainers.neo4j import Neo4jContainer
from testcontainers.postgres import PostgresContainer

PGVECTOR_IMAGE = "pgvector/pgvector:pg16"
NEO4J_IMAGE = "neo4j:5-community"


def _skip_if_no_docker() -> None:
    """Skip the calling fixture if the docker daemon isn't reachable."""
    try:
        import docker  # type: ignore[import-untyped]

        docker.from_env().ping()
    except Exception as exc:
        pytest.skip(f"docker unavailable: {exc}", allow_module_level=False)


@pytest.fixture(scope="session")
def postgres_pgvector() -> Iterator[PostgresContainer]:
    """Postgres + pgvector container, session-scoped.

    Boots `pgvector/pgvector:pg16`. The CREATE EXTENSION + table schema is
    the caller's responsibility — fixtures stay minimal so each test owns
    its data shape.
    """
    _skip_if_no_docker()
    with PostgresContainer(PGVECTOR_IMAGE, driver=None) as container:
        yield container


@pytest.fixture(scope="session")
def neo4j() -> Iterator[Neo4jContainer]:  # type: ignore[explicit-any]
    """Neo4j 5 community container, session-scoped.

    Graphiti requires Neo4j 5+. The container exposes bolt + http; tests
    derive the bolt URL via container.get_connection_url().
    """
    _skip_if_no_docker()
    with Neo4jContainer(NEO4J_IMAGE) as container:
        yield container
