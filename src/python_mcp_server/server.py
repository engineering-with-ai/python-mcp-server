"""MCP server with FastMCP."""

from typing import Optional

from mcp.server.fastmcp import FastMCP

from .clients import Embedder, GraphitiClient, RAGClient
from .config import Config, load_config
from .models import SearchResults, Document, FactEvidence, CombinedResults


def create_server(  # type: ignore[explicit-any]
    config: Optional[Config] = None,
    neo4j_password: Optional[str] = None,
    postgres_password: Optional[str] = None,
    openai_api_key: Optional[str] = None,
) -> FastMCP:
    """Create and configure the MCP server.

    Args:
        config: Optional configuration object. Loads from cfg.yml if not given.
        neo4j_password: Optional Neo4j password. Falls back to NEO4J_PASSWORD env.
        postgres_password: Optional Postgres password. Falls back to POSTGRES_PASSWORD env.
        openai_api_key: Optional OpenAI key. Falls back to OPENAI_API_KEY env.

    Returns:
        Configured FastMCP server instance.
    """
    server_config = config or load_config()
    embedder = Embedder(
        model=server_config.postgres.embedding_model, api_key=openai_api_key
    )

    mcp = FastMCP("Knowledge Graph MCP Server")

    @mcp.tool()
    async def search_knowledge(query: str, limit: int = 10) -> SearchResults:
        """Search factual knowledge in the Graphiti knowledge graph.

        USE WHEN: You need verified facts, entities, relationships, or structured
        knowledge. Combines semantic search, BM25, and graph traversal.
        """
        client = GraphitiClient(server_config.neo4j, password=neo4j_password)
        try:
            results = await client.search(query, limit=limit)
            return SearchResults(items=results, total=len(results))
        finally:
            await client.close()

    @mcp.tool()
    async def rag_search(query: str, limit: int = 10) -> list[Document]:
        """Search documents using vector similarity for context.

        USE WHEN: You need detailed context, explanations, or source documents.
        The query string is embedded internally; no pre-computed vector needed.
        """
        client = RAGClient(
            server_config.postgres, embedder=embedder, password=postgres_password
        )
        return await client.search(query, limit)

    @mcp.tool()
    async def verify_fact(statement: str, limit: int = 5) -> FactEvidence:
        """Retrieve knowledge-graph evidence relevant to a statement.

        USE WHEN: You want to check a claim against the graph. Returns related
        facts; the caller judges whether they support or contradict the statement.
        No boolean verdict — entailment is the caller's job.
        """
        client = GraphitiClient(server_config.neo4j, password=neo4j_password)
        try:
            evidence = await client.search(statement, limit=limit)
            return FactEvidence(statement=statement, evidence=evidence)
        finally:
            await client.close()

    @mcp.tool()
    async def combined_search(query: str, limit: int = 10) -> CombinedResults:
        """Search both the knowledge graph and document vectors.

        USE WHEN: You need both structured facts and supporting context.
        """
        graph_client = GraphitiClient(server_config.neo4j, password=neo4j_password)
        rag_client = RAGClient(
            server_config.postgres, embedder=embedder, password=postgres_password
        )
        try:
            graph_results = await graph_client.search(query, limit=limit)
            vector_results = await rag_client.search(query, limit)
            return CombinedResults(
                graph_results=SearchResults(
                    items=graph_results, total=len(graph_results)
                ),
                vector_results=vector_results,
                query=query,
            )
        finally:
            await graph_client.close()

    @mcp.resource("knowledge://instructions")
    async def knowledge_instructions() -> str:
        """How to answer accurately using this knowledge base."""
        return """
        HOW TO USE THIS KNOWLEDGE BASE EFFECTIVELY:

        Step 1: ALWAYS verify facts before stating them — call verify_fact
                and judge the returned evidence yourself.
        Step 2: Use search_knowledge for entities and relationships.
        Step 3: Use rag_search for detailed explanations.
        Step 4: Combine results for complete answers.
        Step 5: Cite your sources clearly: [Graph] or [Documents].
        Step 6: If evidence is absent or weak, say so — do not guess.
        """

    @mcp.resource("knowledge://examples")
    async def example_queries() -> str:
        """Example patterns for effective knowledge retrieval."""
        return """
        EXAMPLE USAGE PATTERNS:

        1. "What is the relationship between X and Y?"
           → search_knowledge("X Y relationship")

        2. "Explain how Z works"
           → search_knowledge("Z") + rag_search("how Z works")

        3. "Is it true that <claim>?"
           → verify_fact("<claim>"), then judge evidence

        4. Complex question needing both facts and context:
           → combined_search(query)
        """

    @mcp.prompt("answer_with_verification")
    async def answer_with_verification() -> str:
        """Template for answering with evidence-grounded verification."""
        return """
        To answer any question accurately:

        1. IDENTIFY key entities and claims.
        2. For each claim call verify_fact() and inspect the evidence.
        3. Use search_knowledge() for structured facts.
        4. Use rag_search() for supporting documents.
        5. SYNTHESIZE with citations [Graph] or [Documents].
        6. State clearly when evidence is missing rather than guessing.
        """

    return mcp
