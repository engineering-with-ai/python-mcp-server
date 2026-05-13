"""Graphiti knowledge graph client for structured fact retrieval."""

import logging
import os
from typing import Optional
from urllib.parse import urlsplit, urlunsplit

from graphiti_core import Graphiti
from graphiti_core.driver.neptune_driver import NeptuneDriver

from ..models import SearchResult, EntityMetadata

log = logging.getLogger(__name__)


class GraphitiClient:
    """Client for Graphiti knowledge graph operations."""

    def __init__(self, graphiti: Graphiti) -> None:
        """Wrap an already-constructed Graphiti instance.

        Prefer GraphitiClient.from_env() — direct construction is for tests.
        """
        self.graphiti = graphiti

    @classmethod
    def from_env(cls) -> "GraphitiClient":
        """Pick a backend from env vars.

        GRAPH_URL → Neo4j (commercial/ISO; Aura URL carries creds).
        NEPTUNE_HOST + AOSS_HOST → Amazon Neptune + OpenSearch Serverless (defense).
        """
        graph_url = os.environ.get("GRAPH_URL")
        if graph_url:
            # Reason: neo4j-python rejects URL-embedded creds (ConfigurationError),
            # so we strip user:pass out of GRAPH_URL and hand them to Graphiti
            # via the dedicated kwargs.
            uri, user, password = _split_neo4j_url(graph_url)
            return cls(Graphiti(uri=uri, user=user, password=password))

        neptune_host = os.environ.get("NEPTUNE_HOST")
        aoss_host = os.environ.get("AOSS_HOST")
        if neptune_host and aoss_host:
            driver = NeptuneDriver(
                host=f"neptune-db://{neptune_host}",
                aoss_host=aoss_host,
            )
            return cls(Graphiti(graph_driver=driver))

        raise RuntimeError(
            "no graph backend configured — set GRAPH_URL (commercial/ISO) "
            "or NEPTUNE_HOST + AOSS_HOST (defense)"
        )

    async def search(
        self, query: str, center_node_uuid: Optional[str] = None, limit: int = 10
    ) -> list[SearchResult]:
        """Hybrid search combining semantic, BM25, and graph traversal.

        Raises the underlying Graphiti error on failure — no silent empty list.
        """
        try:
            raw_results = await self.graphiti.search(
                query=query, center_node_uuid=center_node_uuid
            )
        except Exception:
            log.exception("graphiti search failed (query=%r)", query)
            raise
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


def _split_neo4j_url(url: str) -> tuple[str, Optional[str], Optional[str]]:
    """Extract (clean_uri, user, password) from a Neo4j URL.

    Aura-style packaging puts creds in the URL for env-var convenience;
    neo4j-python won't accept them there, so we strip them out.
    """
    parts = urlsplit(url)
    host = parts.hostname or ""
    netloc = f"{host}:{parts.port}" if parts.port else host
    clean = urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))
    return clean, parts.username, parts.password
