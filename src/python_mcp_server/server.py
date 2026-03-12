"""MCP server with FastMCP."""

from typing import Optional

from mcp.server.fastmcp import FastMCP

from .clients import GraphitiClient, RAGClient
from .config import Config, load_config
from .models import SearchResults, Document, FactVerification, CombinedResults


def create_server(  # type: ignore[explicit-any]
    config: Optional[Config] = None,
    neo4j_password: Optional[str] = None,
    postgres_url: Optional[str] = None,
) -> FastMCP:
    """Create and configure the MCP server.

    Args:
        config: Optional configuration object. If not provided, loads from cfg.yml.
        neo4j_password: Optional Neo4j password. Falls back to NEO4J_PASSWORD env var.
        postgres_url: Optional PostgreSQL URL. Falls back to POSTGRES_URL env var.

    Returns:
        Configured FastMCP server instance

    Example:
        >>> # Using cfg.yml and env vars
        >>> server = create_server()

        >>> # Explicitly passing configuration
        >>> from python_mcp_server.config import Config, Neo4jConfig, PostgresConfig, LogLevel
        >>> cfg = Config(
        ...     log_level=LogLevel.INFO,
        ...     neo4j=Neo4jConfig(uri="bolt://localhost:7687", user="neo4j", database="neo4j"),
        ...     postgres=PostgresConfig(embeddings_table="my_table")
        ... )
        >>> server = create_server(config=cfg, neo4j_password="secret", postgres_url="postgresql://...")

    """
    # Use provided config or load from cfg.yml
    server_config = config or load_config()

    mcp = FastMCP("Knowledge Graph MCP Server")

    @mcp.tool()
    async def search_knowledge(query: str, limit: int = 10) -> SearchResults:
        """Search factual knowledge using Graphiti's knowledge graph.

        USE THIS WHEN: You need verified facts, entities, relationships, or structured knowledge.
        Returns: Entities with their properties and relationships from the knowledge graph.

        This tool combines:
        - Semantic search (understanding meaning)
        - BM25 keyword matching (exact terms)
        - Graph traversal (connected facts)

        Example queries:
        - "What companies did John Smith work for?"
        - "Show me all products related to machine learning"
        - "Find connections between Tesla and battery technology"
        """
        client = GraphitiClient(server_config.neo4j, password=neo4j_password)
        try:
            results = await client.search(query, limit=limit)
            return SearchResults(items=results, total=len(results))
        finally:
            await client.close()

    @mcp.tool()
    async def rag_search(
        query_embedding: list[float], limit: int = 10
    ) -> list[Document]:
        """Search documents using vector similarity for context and supporting information.

        USE THIS WHEN: You need detailed context, explanations, or source documents.
        Returns: Full document chunks with metadata and sources.

        Requires: Generate embedding for your query first.
        Returns documents ranked by semantic similarity.

        Best for:
        - Finding detailed explanations
        - Getting source quotations
        - Retrieving procedural information
        """
        client = RAGClient(server_config.postgres, postgres_url=postgres_url)
        results = await client.vector_search(query_embedding, limit)
        return results

    @mcp.tool()
    async def verify_fact(statement: str) -> FactVerification:
        """Verify a specific fact against the knowledge graph.

        USE THIS FIRST: Before making any factual claims, verify them here.
        Returns: Whether the fact exists in the graph, with supporting evidence.

        This prevents hallucination by checking claims against verified knowledge.

        Example:
        - Statement: "Tesla was founded in 2003"
        - Returns: verified=True, evidence=[nodes and relationships]
        """
        client = GraphitiClient(server_config.neo4j, password=neo4j_password)
        try:
            verification = await client.verify_fact(statement)
            return verification
        finally:
            await client.close()

    @mcp.tool()
    async def combined_search(
        query: str, query_embedding: list[float], limit: int = 10
    ) -> CombinedResults:
        """Search both Graphiti graph and RAG vectors for comprehensive results.

        USE THIS WHEN: You need both verified facts and supporting context.
        Returns: Combined results from knowledge graph and documents.

        This is your most powerful tool for answering complex questions that need
        both structured facts and detailed explanations.
        """
        # Search knowledge graph
        graph_client = GraphitiClient(server_config.neo4j, password=neo4j_password)
        rag_client = RAGClient(server_config.postgres, postgres_url=postgres_url)

        try:
            graph_results = await graph_client.search(query, limit=limit)
            vector_results = await rag_client.vector_search(query_embedding, limit)

            search_results = SearchResults(
                items=graph_results, total=len(graph_results)
            )

            return CombinedResults(
                graph_results=search_results, vector_results=vector_results, query=query
            )
        finally:
            await graph_client.close()

    @mcp.resource("knowledge://instructions")
    async def knowledge_instructions() -> str:
        """CRITICAL: How to answer questions accurately using this knowledge base.

        ALWAYS follow these steps to prevent hallucination:

        1. VERIFY FACTS FIRST: Use 'verify_fact' before making any factual claims
        2. Use 'search_knowledge' for structured facts and relationships from the graph
        3. Use 'rag_search' for detailed context and explanations from documents
        4. Use 'combined_search' for comprehensive answers needing both facts and context
        5. If no results found, say "I don't have information about that" - NEVER guess
        6. Always cite which tool provided the information: [Graph] or [Documents]

        TOOL SELECTION GUIDE:
        - Factual questions: verify_fact → search_knowledge
        - Need explanations: search_knowledge → rag_search
        - Complex questions: combined_search
        - Checking claims: verify_fact

        The knowledge graph contains VERIFIED FACTS.
        The documents contain SUPPORTING CONTEXT.
        Use both to provide accurate, grounded answers.
        """
        return """
        HOW TO USE THIS KNOWLEDGE BASE EFFECTIVELY:
        
        Step 1: ALWAYS verify facts before stating them
        Step 2: Use search_knowledge for entities and relationships
        Step 3: Use rag_search for detailed explanations
        Step 4: Combine results for complete answers
        Step 5: Cite your sources clearly
        Step 6: Admit when information is not available
        
        This approach prevents hallucination and ensures accuracy.
        """

    @mcp.resource("knowledge://examples")
    async def example_queries() -> str:
        """Example patterns for effective knowledge retrieval."""
        return """
        EXAMPLE USAGE PATTERNS:
        
        1. User asks: "What is the relationship between X and Y?"
           Approach: 
           - verify_fact("X is related to Y")
           - search_knowledge("X relationship Y")
           - Result: Verified facts with citations [Graph]
        
        2. User asks: "Explain how Z works"
           Approach:
           - search_knowledge("Z") for basic facts
           - rag_search(embedding("how Z works")) for detailed explanation
           - Result: Facts [Graph] + Context [Documents]
        
        3. User asks: "Is it true that..."
           Approach:
           - verify_fact(statement) FIRST
           - If verified, provide evidence
           - If not verified, state clearly
           - Result: Fact-checked response
        
        4. Complex question needing both facts and context:
           Approach:
           - combined_search(query, embedding) 
           - Result: Comprehensive answer with sources
        """

    @mcp.prompt("answer_with_verification")
    async def answer_with_verification() -> str:
        """Template for answering questions with proper fact verification."""
        return """
        To answer any question accurately:
        
        1. IDENTIFY key entities and claims in the question
        2. VERIFY each factual claim using verify_fact()
        3. SEARCH for additional context using search_knowledge()
        4. RETRIEVE supporting documents using rag_search() if needed
        5. SYNTHESIZE answer using:
           - Verified facts from knowledge graph [Graph]
           - Supporting context from documents [Documents]
        6. CITE sources clearly for each piece of information
        7. STATE clearly if information is not available rather than guessing
        
        This process ensures accuracy and prevents hallucination.
        """

    return mcp
