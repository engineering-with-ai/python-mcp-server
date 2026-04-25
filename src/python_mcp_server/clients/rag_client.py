"""RAG client for pgvector similarity search."""

import os
from typing import Final, Optional

import asyncpg
from pgvector.asyncpg import register_vector

from ..config import PostgresConfig
from ..models import Document, DocumentMetadata
from .embedder import Embedder

RRF_K: Final[int] = 60


def rrf_fuse(rankings: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion over multiple rankings.

    Args:
        rankings: Each inner list is an ordered list of doc ids (best first).
        k: RRF constant. Default 60 — published industry-standard, parameter-free.

    Returns:
        (doc_id, score) pairs sorted by score descending.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


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
        """Hybrid retrieval: cosine + BM25 fused via Reciprocal Rank Fusion.

        Runs two rankings against the embeddings table — cosine over the
        embedding column and BM25 over the content_tsv column — then fuses
        via RRF (k=60). Exact-term matches (protocol field names, enum values,
        IDs) come through the BM25 leg that pure cosine would miss.
        """
        query_embedding = await self.embedder.embed(query)
        conn = await asyncpg.connect(self.db_url)
        try:
            await register_vector(conn)
            # Reason: table_name is from cfg.yml, not user input - safe from injection
            cosine_sql = f"""
                SELECT id, title, content, book, section_level, analysis_relevance,
                       effective_from, effective_until, normative
                FROM {self.table_name}
                ORDER BY embedding <=> $1::vector
                LIMIT $2
                """  # noqa: S608
            bm25_sql = f"""
                SELECT id, title, content, book, section_level, analysis_relevance,
                       effective_from, effective_until, normative
                FROM {self.table_name}
                WHERE content_tsv @@ plainto_tsquery('english', $1)
                ORDER BY ts_rank_cd(content_tsv, plainto_tsquery('english', $1)) DESC
                LIMIT $2
                """  # noqa: S608
            cosine_rows = await conn.fetch(cosine_sql, query_embedding, limit)
            bm25_rows = await conn.fetch(bm25_sql, query, limit)
        finally:
            await conn.close()

        docs: dict[str, Document] = {}
        for row in list(cosine_rows) + list(bm25_rows):
            doc_id = str(row["id"])
            if doc_id in docs:
                continue
            docs[doc_id] = Document(
                id=doc_id,
                title=str(row.get("title", "")),
                content=str(row["content"]),
                metadata=_build_metadata(row),
                similarity_score=0.0,
            )

        cosine_ids = [str(row["id"]) for row in cosine_rows]
        bm25_ids = [str(row["id"]) for row in bm25_rows]
        fused = rrf_fuse([cosine_ids, bm25_ids])

        ordered: list[Document] = []
        for doc_id, score in fused[:limit]:
            doc = docs[doc_id]
            doc.similarity_score = score
            ordered.append(doc)
        return ordered

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
                       effective_from, effective_until, normative,
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
                metadata=_build_metadata(row),
                similarity_score=float(row["distance"]),
            )
            for row in rows
        ]


def _build_metadata(row: "asyncpg.Record") -> DocumentMetadata:
    """Project a knowledge-table row into DocumentMetadata.

    Temporal columns are TIMESTAMPTZ; surface them as ISO strings so the LLM
    can reason about enforceability without a date library on the client side.
    `normative` flows through as bool. `effective_until = NULL` means still
    in force and is preserved as None for the LLM to interpret.
    """
    eff_from = row.get("effective_from")
    eff_until = row.get("effective_until")
    return DocumentMetadata(
        book=str(row.get("book", "")),
        section_level=str(row.get("section_level", "")),
        analysis_relevance=str(row.get("analysis_relevance", "")),
        effective_from=eff_from.isoformat() if eff_from is not None else None,
        effective_until=eff_until.isoformat() if eff_until is not None else None,
        normative=row.get("normative"),
    )
