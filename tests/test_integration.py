"""Integration tests for MCP server with Graphiti and RAG."""

import pytest
from unittest.mock import AsyncMock, patch

from src.python_mcp_server.clients.graphiti_client import GraphitiClient
from src.python_mcp_server.clients.rag_client import RAGClient
from src.python_mcp_server.config import Config, Neo4jConfig, PostgresConfig, LogLevel
from src.python_mcp_server.server import create_server


class TestGraphitiClient:
    """Test Graphiti client functionality."""

    @pytest.mark.asyncio
    async def test_graphiti_client_search_returns_results(self) -> None:
        """Test that Graphiti client can perform search."""
        # Arrange
        test_config = Neo4jConfig(
            uri="bolt://localhost:7687", user="neo4j", database="neo4j"
        )
        with (
            patch(
                "src.python_mcp_server.clients.graphiti_client.Graphiti"
            ) as mock_graphiti_class,
            patch(
                "src.python_mcp_server.clients.graphiti_client.os.getenv"
            ) as mock_getenv,
        ):
            mock_getenv.return_value = "test_password"
            mock_graphiti = AsyncMock()
            mock_graphiti_class.return_value = mock_graphiti
            # Create a mock object with proper attributes
            mock_result = type("MockResult", (), {})()
            mock_result.uuid = "test"
            mock_result.id = "test"
            mock_result.fact = "test result"
            mock_result.content = "test result"
            mock_result.score = 0.9

            mock_graphiti.search.return_value = [mock_result]

            client = GraphitiClient(test_config)

            # Act
            results = await client.search("test query")

            # Assert
            assert len(results) == 1
            assert results[0].id == "test"
            assert results[0].content == "test result"
            mock_graphiti.search.assert_called_once_with(
                query="test query", center_node_uuid=None
            )


class TestRAGClient:
    """Test RAG client functionality."""

    @pytest.mark.asyncio
    async def test_rag_client_vector_search_returns_results(self) -> None:
        """Test that RAG client can perform vector search."""
        # Arrange
        test_config = PostgresConfig(embeddings_table="energy_embeddings")
        with (
            patch(
                "src.python_mcp_server.clients.rag_client.asyncpg.connect"
            ) as mock_connect,
            patch("src.python_mcp_server.clients.rag_client.os.getenv") as mock_getenv,
        ):
            mock_getenv.return_value = "postgresql://test:test@localhost:5432/test"
            mock_conn = AsyncMock()
            mock_connect.return_value = mock_conn
            mock_conn.fetch.return_value = [
                {
                    "id": "doc1",
                    "content": "test document",
                    "metadata": {},
                    "distance": 0.1,
                }
            ]

            client = RAGClient(test_config)
            query_embedding = [0.1, 0.2, 0.3]

            # Act
            results = await client.vector_search(query_embedding)

            # Assert
            assert len(results) == 1
            assert results[0].id == "doc1"
            assert results[0].content == "test document"
            mock_conn.fetch.assert_called_once()

    @pytest.mark.asyncio
    async def test_rag_client_queries_energy_embeddings_table(self) -> None:
        """Test that RAG client queries energy_embeddings table with proper schema."""
        # Arrange
        test_config = PostgresConfig(embeddings_table="energy_embeddings")
        with (
            patch(
                "src.python_mcp_server.clients.rag_client.asyncpg.connect"
            ) as mock_connect,
            patch("src.python_mcp_server.clients.rag_client.os.getenv") as mock_getenv,
        ):
            mock_getenv.return_value = "postgresql://test:test@localhost:5432/test"
            mock_conn = AsyncMock()
            mock_connect.return_value = mock_conn
            mock_conn.fetch.return_value = [
                {
                    "id": 1,
                    "title": "Power Systems Basics - Grid Fundamentals",
                    "content": "The electric grid operates at multiple voltage levels",
                    "book": "Power Systems Basics",
                    "section_level": "h2",
                    "analysis_relevance": "high",
                    "distance": 0.05,
                }
            ]

            client = RAGClient(test_config)
            query_embedding = [0.1] * 1536

            # Act
            results = await client.vector_search(query_embedding, limit=5)

            # Assert
            assert len(results) == 1
            assert (
                results[0].content
                == "The electric grid operates at multiple voltage levels"
            )
            assert results[0].similarity_score == 0.05

            # Verify the query uses energy_embeddings table
            call_args = mock_conn.fetch.call_args
            query_sql = call_args[0][0]
            assert "energy_embeddings" in query_sql


class TestMCPServer:
    """Test MCP server functionality."""

    def test_create_server_returns_fastmcp_instance(self) -> None:
        """Test that create_server returns a FastMCP instance."""
        # Act
        server = create_server()

        # Assert
        assert server is not None
        assert str(type(server)) == "<class 'mcp.server.fastmcp.server.FastMCP'>"
        # Check what methods are available
        print(
            "FastMCP methods:",
            [method for method in dir(server) if not method.startswith("_")],
        )

    @pytest.mark.asyncio
    async def test_search_knowledge_tool_exists(self) -> None:
        """Test that search_knowledge tool is registered with server."""
        # Arrange
        server = create_server()

        # Act
        tools = await server.list_tools()

        # Assert
        tool_names = [tool.name for tool in tools]
        assert "search_knowledge" in tool_names

        # Check the tool has the expected description
        search_tool = next(tool for tool in tools if tool.name == "search_knowledge")
        description = search_tool.description or ""
        assert "USE THIS WHEN" in description
        assert "verified facts" in description

    @pytest.mark.asyncio
    async def test_search_knowledge_tool_execution(self) -> None:
        """Test that search_knowledge tool can be called and returns correct format."""
        # Arrange
        with (
            patch(
                "src.python_mcp_server.clients.graphiti_client.Graphiti"
            ) as mock_graphiti_class,
            patch(
                "src.python_mcp_server.clients.graphiti_client.os.getenv"
            ) as mock_getenv,
        ):
            mock_getenv.return_value = "test_password"
            mock_graphiti = AsyncMock()
            mock_graphiti_class.return_value = mock_graphiti
            # Create a mock object with proper attributes
            mock_result = type("MockResult", (), {})()
            mock_result.uuid = "test_id"
            mock_result.id = "test_id"
            mock_result.fact = "test content"
            mock_result.content = "test content"
            mock_result.score = 0.9

            mock_graphiti.search.return_value = [mock_result]

            server = create_server()

            # Act
            _content, result_data = await server.call_tool(
                "search_knowledge", {"query": "test query", "limit": 5}
            )

            # Assert - FastMCP returns tuple of (content, result_dict)
            # Runtime behavior returns dict despite type annotations
            assert result_data is not None
            assert isinstance(result_data, dict)  # type: ignore[unreachable]
            assert "items" in result_data  # type: ignore[unreachable]
            assert "total" in result_data
            assert "source" in result_data
            assert result_data["source"] == "knowledge_graph"
            assert len(result_data["items"]) == 1
            # Access nested dict structure for MCP result data
            first_item = result_data["items"][0]
            assert first_item["id"] == "test_id"
            assert first_item["content"] == "test content"

    @pytest.mark.asyncio
    async def test_knowledge_instructions_resource_exists(self) -> None:
        """Test that knowledge instructions resource is available."""
        # Arrange
        server = create_server()

        # Act
        resources = await server.list_resources()

        # Assert
        resource_uris = [str(resource.uri) for resource in resources]
        assert "knowledge://instructions" in resource_uris

        # Test reading the resource
        content_iterable = await server.read_resource("knowledge://instructions")
        # Handle the resource content properly - it's an Iterable[ReadResourceContents]
        content_text = ""
        content_items = list(content_iterable)
        if content_items:
            first_content = content_items[0]
            if hasattr(first_content, "text"):
                content_text = first_content.text
            elif hasattr(first_content, "content"):
                content = first_content.content
                content_text = (
                    content if isinstance(content, str) else content.decode("utf-8")
                )
            else:
                content_text = str(first_content)

        assert "ALWAYS verify facts" in content_text
        assert "prevents hallucination" in content_text

    def test_create_server_with_explicit_config(self) -> None:
        """Test that create_server accepts explicit config parameters."""
        # Arrange
        test_config = Config(
            log_level=LogLevel.INFO,
            neo4j=Neo4jConfig(
                uri="bolt://test:7687", user="test_user", database="test_db"
            ),
            postgres=PostgresConfig(embeddings_table="test_table"),
        )

        # Act
        server = create_server(
            config=test_config,
            neo4j_password="test_password",  # noqa: S106
            postgres_url="postgresql://test:test@localhost:5432/test",
        )

        # Assert
        assert server is not None
        assert str(type(server)) == "<class 'mcp.server.fastmcp.server.FastMCP'>"

    @pytest.mark.asyncio
    async def test_create_server_with_explicit_config_can_call_tools(self) -> None:
        """Test that server created with explicit config can execute tools."""
        # Arrange
        test_config = Config(
            log_level=LogLevel.INFO,
            neo4j=Neo4jConfig(
                uri="bolt://test:7687", user="test_user", database="test_db"
            ),
            postgres=PostgresConfig(embeddings_table="test_table"),
        )

        with (
            patch(
                "src.python_mcp_server.clients.graphiti_client.Graphiti"
            ) as mock_graphiti_class,
        ):
            mock_graphiti = AsyncMock()
            mock_graphiti_class.return_value = mock_graphiti
            mock_result = type("MockResult", (), {})()
            mock_result.uuid = "test_id"
            mock_result.id = "test_id"
            mock_result.fact = "test content"
            mock_result.content = "test content"
            mock_result.score = 0.9

            mock_graphiti.search.return_value = [mock_result]

            # Act
            server = create_server(
                config=test_config,
                neo4j_password="test_password",  # noqa: S106
                postgres_url="postgresql://test:test@localhost:5432/test",
            )
            _content, result_data = await server.call_tool(
                "search_knowledge", {"query": "test query", "limit": 5}
            )

            # Assert
            # Runtime behavior returns dict despite type annotations
            assert result_data is not None
            assert isinstance(result_data, dict)  # type: ignore[unreachable]
            assert "items" in result_data  # type: ignore[unreachable]
            assert len(result_data["items"]) == 1
