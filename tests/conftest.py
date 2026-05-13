"""Pytest auto-discovery shim: surface container fixtures to all tests/."""

from tests.fixtures.containers import neo4j, postgres_pgvector

__all__ = ["neo4j", "postgres_pgvector"]
