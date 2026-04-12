"""RAG client for pgvector similarity search."""

import os
from typing import Optional

import asyncpg
from pgvector.asyncpg import register_vector

from ..config import PostgresConfig
from ..models import Document, DocumentMetadata
from .embedder import Embedder


class RAGClient:
    """Client for RAG vector similarity search."""

    def __init__(
        self,
        config: PostgresConfig,
        embedder: Embedder,
        password: Optional[str] = None,
    ) -> None:
        """Initialize RAG client.

        Args:
            config: PostgreSQL configuration (host, port, db, user, table, model)
            embedder: Injected embedder for query string → vector
            password: Optional password. Falls back to POSTGRES_PASSWORD env var.
        """
        pwd = password or os.getenv("POSTGRES_PASSWORD", "")
        self.db_url = (
            f"postgresql://{config.user}:{pwd}@{config.host}:{config.port}/"
            f"{config.database}"
        )
        self.table_name = config.embeddings_table
        self.embedder = embedder

    async def search(self, query: str, limit: int = 10) -> list[Document]:
        """Embed a string query and run vector search."""
        query_embedding = await self.embedder.embed(query)
        return await self.vector_search(query_embedding, limit)

    async def vector_search(
        self, query_embedding: list[float], limit: int = 10
    ) -> list[Document]:
        """Vector similarity search using a pre-computed embedding."""
        conn = await asyncpg.connect(self.db_url)
        try:
            await register_vector(conn)
            # Reason: table_name is from cfg.yml, not user input - safe from injection
            sql = f"""
                SELECT id, title, content, book, section_level, analysis_relevance,
                       embedding <=> $1::vector as distance
                FROM {self.table_name}
                ORDER BY distance
                LIMIT $2
                """  # noqa: S608
            rows = await conn.fetch(sql, query_embedding, limit)
        finally:
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
            for row in rows
        ]
