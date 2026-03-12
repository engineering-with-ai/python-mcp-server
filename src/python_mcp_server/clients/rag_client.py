"""RAG client for pgvector similarity search."""

import os
from typing import Optional

import asyncpg
from pgvector.asyncpg import register_vector

from ..config import PostgresConfig
from ..models import Document, DocumentMetadata


class RAGClient:
    """Client for RAG vector similarity search."""

    def __init__(
        self, config: PostgresConfig, postgres_url: Optional[str] = None
    ) -> None:
        """Initialize RAG client with provided configuration.

        Args:
            config: PostgreSQL configuration settings (table name, etc.)
            postgres_url: Optional PostgreSQL connection URL. Falls back to POSTGRES_URL env var if not provided.

        """
        # Use provided URL or fall back to environment variable
        self.db_url = postgres_url or os.getenv(
            "POSTGRES_URL", "postgresql://user:password@localhost:5432/knowledge"
        )
        self.table_name = config.embeddings_table

    async def vector_search(
        self, query_embedding: list[float], limit: int = 10
    ) -> list[Document]:
        """Simple, elegant vector similarity search."""
        try:
            conn = await asyncpg.connect(self.db_url)
            await register_vector(conn)

            # Reason: table_name is from cfg.yml, not user input - safe from injection
            query = f"""
                SELECT id, title, content, book, section_level, analysis_relevance,
                       embedding <=> $1::vector as distance
                FROM {self.table_name}
                ORDER BY distance
                LIMIT $2
                """  # noqa: S608

            results = await conn.fetch(
                query,
                query_embedding,
                limit,
            )
            await conn.close()

            return [
                Document(
                    id=str(row["id"]),
                    title=str(row.get("title", "")),
                    content=str(row["content"]),
                    metadata=DocumentMetadata(
                        book=str(row.get("book", "")),
                        section_level=str(row.get("section_level", "")),
                        analysis_relevance=str(row.get("analysis_relevance", "")),
                    ),
                    similarity_score=float(row["distance"]),
                )
                for row in results
            ]
        except Exception:
            # Reason: Return empty list for graceful degradation
            return []
