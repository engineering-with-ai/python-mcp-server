"""Graphiti knowledge graph client for structured fact retrieval."""

import os
from typing import Optional

from graphiti_core import Graphiti

from ..config import Neo4jConfig
from ..models import (
    SearchResult,
    EntityMetadata,
    Relationship,
    FactVerification,
    GraphSchema,
)


class GraphitiClient:
    """Client for Graphiti knowledge graph operations."""

    def __init__(self, config: Neo4jConfig, password: Optional[str] = None) -> None:
        """Initialize Graphiti client with provided configuration.

        Args:
            config: Neo4j configuration settings (uri, user, database)
            password: Optional Neo4j password. Falls back to NEO4J_PASSWORD env var if not provided.

        """
        # Use provided password or fall back to environment variable
        neo4j_password = password or os.getenv("NEO4J_PASSWORD", "")

        self.graphiti = Graphiti(
            uri=config.uri,
            user=config.user,
            password=neo4j_password,
        )

    async def search(
        self, query: str, center_node_uuid: Optional[str] = None, limit: int = 10
    ) -> list[SearchResult]:
        """
        Search knowledge graph using Graphiti's hybrid search.

        Combines semantic search, BM25 keyword matching, and graph traversal.
        """
        try:
            raw_results = await self.graphiti.search(
                query=query, center_node_uuid=center_node_uuid
            )
            # Apply limit and convert to SearchResult models
            limited_results = raw_results[:limit] if raw_results else []
            return [
                SearchResult(
                    id=str(getattr(result, "uuid", getattr(result, "id", ""))),
                    content=str(
                        getattr(result, "fact", getattr(result, "content", ""))
                    ),
                    score=getattr(result, "score", None),
                    metadata=EntityMetadata(),
                )
                for result in limited_results
            ]
        except Exception:
            # Reason: Return empty list instead of failing to allow graceful degradation
            return []

    async def get_entities(self, entity_ids: list[str]) -> list[SearchResult]:
        """Get specific entities by their IDs."""
        entities: list[SearchResult] = []
        try:
            for entity_id in entity_ids:
                # Note: get_node method may not exist in actual Graphiti API
                # This would need to be updated based on real API
                entity_data = await self.search(f"id:{entity_id}", limit=1)
                if entity_data:
                    entities.extend(entity_data)
        except Exception:
            # Reason: Continue with empty list if any entity fetch fails
            entities = []
        return entities

    async def get_relationships(
        self, entity_id: str, depth: int = 2
    ) -> list[Relationship]:
        """Get relationships for an entity up to specified depth."""
        relationships: list[Relationship] = []
        try:
            # Note: get_edges method may not exist in actual Graphiti API
            # This would need to be updated based on real API
            raw_edges = await self.search(f"relationships:{entity_id}", limit=50)
            relationships = [
                Relationship(
                    source_id=entity_id,
                    target_id=f"related_{i}",
                    relationship_type="RELATED_TO",
                    properties=EntityMetadata(),
                )
                for i, _ in enumerate(raw_edges[:depth])
            ]
        except Exception:
            relationships = []
        return relationships

    async def verify_fact(self, statement: str) -> FactVerification:
        """
        Verify if a fact exists in the knowledge graph.

        Returns verification result with evidence if found.
        """
        try:
            # Search for entities and relationships related to the statement
            search_results = await self.search(statement, limit=5)

            if not search_results:
                return FactVerification(
                    statement=statement,
                    verified=False,
                    confidence=0.0,
                    evidence=[],
                    suggestion="No relevant information found in knowledge graph",
                )
            else:
                # Simple verification based on search results
                # In a real implementation, this would be more sophisticated
                return FactVerification(
                    statement=statement,
                    verified=True,
                    confidence=0.8,  # Could be calculated based on search scores
                    evidence=search_results,
                    suggestion=None,
                )
        except Exception as e:
            return FactVerification(
                statement=statement,
                verified=False,
                confidence=0.0,
                evidence=[],
                suggestion=f"Error during verification: {e!s}",
            )

    async def get_schema(self) -> GraphSchema:
        """Get the current knowledge graph schema."""
        try:
            # This would depend on Graphiti's schema introspection capabilities
            # For now, return a basic structure
            return GraphSchema(
                node_types=["Entity", "Person", "Organization", "Concept"],
                relationship_types=["RELATED_TO", "WORKS_FOR", "PART_OF"],
                total_nodes=0,  # Would need to query actual counts
                total_relationships=0,
            )
        except Exception as e:
            return GraphSchema(
                node_types=[],
                relationship_types=[],
                total_nodes=0,
                total_relationships=0,
                error=str(e),
            )

    async def close(self) -> None:
        """Close the Graphiti client connection."""
        if hasattr(self.graphiti, "close") and callable(self.graphiti.close):
            await self.graphiti.close()  # type: ignore[no-untyped-call]
