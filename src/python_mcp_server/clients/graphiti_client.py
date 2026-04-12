"""Graphiti knowledge graph client for structured fact retrieval."""

import os
from typing import Optional

from graphiti_core import Graphiti

from ..config import Neo4jConfig
from ..models import SearchResult, EntityMetadata


class GraphitiClient:
    """Client for Graphiti knowledge graph operations."""

    def __init__(self, config: Neo4jConfig, password: Optional[str] = None) -> None:
        """Initialize Graphiti client.

        Args:
            config: Neo4j configuration (uri, user, database)
            password: Optional password. Falls back to NEO4J_PASSWORD env var.
        """
        neo4j_password = password or os.getenv("NEO4J_PASSWORD", "")
        self.graphiti = Graphiti(
            uri=config.uri,
            user=config.user,
            password=neo4j_password,
        )

    async def search(
        self, query: str, center_node_uuid: Optional[str] = None, limit: int = 10
    ) -> list[SearchResult]:
        """Hybrid search combining semantic, BM25, and graph traversal.

        Raises the underlying Graphiti error on failure — no silent empty list.
        """
        raw_results = await self.graphiti.search(
            query=query, center_node_uuid=center_node_uuid
        )
        limited = raw_results[:limit] if raw_results else []
        return [
            SearchResult(
                id=str(getattr(r, "uuid", getattr(r, "id", ""))),
                content=str(getattr(r, "fact", getattr(r, "content", ""))),
                score=getattr(r, "score", None),
                metadata=EntityMetadata(),
            )
            for r in limited
        ]

    async def close(self) -> None:
        """Close the Graphiti client connection."""
        if hasattr(self.graphiti, "close") and callable(self.graphiti.close):
            await self.graphiti.close()  # type: ignore[no-untyped-call]
