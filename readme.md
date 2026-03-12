# Python MCP Server 🧠

![](https://img.shields.io/gitlab/pipeline-status/agent-framework/python-mcp-server?branch=main&logo=gitlab)
![](https://gitlab.com/agent-framework/python-mcp-server/badges/main/coverage.svg)
![](https://img.shields.io/badge/3.13.1-gray?logo=python)
![](https://img.shields.io/badge/0.9.15-gray?logo=uv)
![](https://img.shields.io/badge/5.0.0-gray?logo=neo4j)
![](https://img.shields.io/badge/16.0.0-white?logo=postgresql)

A Model Context Protocol (MCP) server that provides AI agents with access to knowledge graphs (via Graphiti) and vector search (via pgvector) for accurate, grounded responses.

## Features

- **🎯 Fact Verification**: Verify claims against knowledge graph before making statements
- **🔍 Hybrid Search**: Combines semantic search, keyword matching, and graph traversal
- **📚 Vector RAG**: Semantic similarity search over document embeddings
- **🧠 Knowledge Graph**: Powered by Graphiti for temporal, relationship-aware knowledge
- **🔧 Dual Distribution**: Works as both library and CLI application
- **⚙️ Flexible Configuration**: Three configuration methods (env vars, cfg.yml, or programmatic)
- **🚫 Anti-Hallucination**: Built-in verification tools to prevent false information

## Quick Start

### Installation

```bash
# Install via pip
pip install python-mcp-server

# Or run directly with uvx
uvx python-mcp-server
```

## Configuration

The server supports **three flexible configuration methods**:

### 1. Environment Variables Only (Recommended for Production)

Set all configuration via environment variables:

```bash
# Required - Secrets
export NEO4J_PASSWORD="your_password"
export POSTGRES_URL="postgresql://user:pass@localhost:5432/knowledge"

# Optional - Configuration (uses defaults if not set)
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_DATABASE="neo4j"
export POSTGRES_TABLE="embeddings"
export LOG_LEVEL="INFO"
export ENV="local"
```

### 2. cfg.yml + Environment Variables (Development)

Create `cfg.yml` for configuration, use env vars for secrets:

```yaml
local:
  log_level: DEBUG
  neo4j:
    uri: bolt://localhost:7687
    user: neo4j
    database: neo4j
  postgres:
    embeddings_table: energy_embeddings

beta:
  log_level: INFO
  neo4j:
    uri: bolt://production:7687
    user: neo4j
    database: neo4j
  postgres:
    embeddings_table: energy_embeddings
```

Set secrets in environment:
```bash
export NEO4J_PASSWORD="your_password"
export POSTGRES_URL="postgresql://user:pass@localhost:5432/knowledge"
export ENV="local"  # or "beta"
```

### 3. Programmatic Configuration (Library Usage)

Pass configuration explicitly when using as a library:

```python
from python_mcp_server import create_server
from python_mcp_server.config import Config, Neo4jConfig, PostgresConfig, LogLevel

# Create explicit configuration
config = Config(
    log_level=LogLevel.INFO,
    neo4j=Neo4jConfig(
        uri="bolt://localhost:7687",
        user="neo4j",
        database="neo4j"
    ),
    postgres=PostgresConfig(
        embeddings_table="my_embeddings"
    )
)

# Create server with explicit config and secrets
server = create_server(
    config=config,
    neo4j_password="your_password",  # Optional, falls back to env
    postgres_url="postgresql://user:pass@localhost:5432/knowledge"  # Optional, falls back to env
)
```

## Usage Examples

### Usage with Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "knowledge-graph": {
      "command": "uvx",
      "args": ["python-mcp-server"],
      "env": {
        "NEO4J_PASSWORD": "your_password",
        "POSTGRES_URL": "postgresql://user:pass@localhost:5432/knowledge",
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USER": "neo4j",
        "NEO4J_DATABASE": "neo4j",
        "POSTGRES_TABLE": "embeddings"
      }
    }
  }
}
```

### Usage with Pydantic-AI

```python
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio

# Option A: Use environment variables
import os
os.environ["NEO4J_PASSWORD"] = "your_password"
os.environ["POSTGRES_URL"] = "postgresql://user:pass@localhost:5432/knowledge"

mcp = MCPServerStdio("uvx", "python-mcp-server")
agent = Agent(toolsets=[mcp])

# Option B: Use explicit configuration (if using as library)
from python_mcp_server import create_server
from python_mcp_server.config import Config, Neo4jConfig, PostgresConfig, LogLevel

config = Config(
    log_level=LogLevel.INFO,
    neo4j=Neo4jConfig(uri="bolt://localhost:7687", user="neo4j", database="neo4j"),
    postgres=PostgresConfig(embeddings_table="embeddings")
)
server = create_server(config=config, neo4j_password="pwd", postgres_url="postgresql://...")

# Agent now has access to knowledge graph and vector search
result = await agent.run("What is the relationship between Tesla and battery technology?")
```

## Available Tools

### 🔍 `search_knowledge`
Search the knowledge graph for verified facts and relationships.
- **Use when**: Need factual information, entities, relationships
- **Returns**: Structured data from knowledge graph
- **Best for**: "Who", "what", "when" questions

### 📖 `rag_search` 
Search documents using vector similarity.
- **Use when**: Need detailed explanations or context
- **Returns**: Relevant document chunks with similarity scores
- **Best for**: "How", "why", "explain" questions

### ✅ `verify_fact`
Verify specific claims against the knowledge graph.
- **Use when**: Checking factual accuracy before making claims
- **Returns**: Verification status with supporting evidence
- **Best for**: Fact-checking statements

### 🔄 `combined_search`
Search both knowledge graph and documents simultaneously.
- **Use when**: Need comprehensive answers with both facts and context
- **Returns**: Results from both knowledge sources
- **Best for**: Complex questions requiring multiple information types

## Resources & Guidance

The server provides built-in resources to guide AI agents:

- `knowledge://instructions` - Step-by-step usage guide
- `knowledge://examples` - Query patterns and examples

## Database Schema

Your PostgreSQL table should have the following schema for vector search:

```sql
CREATE TABLE embeddings (
    id SERIAL PRIMARY KEY,
    title TEXT,
    content TEXT NOT NULL,
    book TEXT,
    section_level TEXT,
    analysis_relevance TEXT,
    embedding vector(1536)  -- pgvector extension required
);

CREATE INDEX ON embeddings USING ivfflat (embedding vector_cosine_ops);
```

## Development

### Setup
```bash
git clone <repo>
cd python-mcp-server
uv sync --dev

# Copy and configure environment variables
cp template-secrets.env .env
# Edit .env with your database credentials
```

### Testing
```bash
# Run tests
uv run pytest

# Run with coverage
uv run poe cover

# Run all checks (linting, type checking, security)
uv run poe checks
```

### Running Locally
```bash
# Option 1: Using cfg.yml + environment variables (recommended for development)
export NEO4J_PASSWORD="your_password"
export POSTGRES_URL="postgresql://user:pass@localhost:5432/knowledge"
export ENV="local"  # Uses cfg.yml local configuration
uv run python-mcp-server

# Option 2: Using environment variables only
export NEO4J_PASSWORD="your_password"
export POSTGRES_URL="postgresql://user:pass@localhost:5432/knowledge"
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_DATABASE="neo4j"
export POSTGRES_TABLE="embeddings"
uv run python-mcp-server
```

## Architecture

```
python-mcp-server/
├── src/python_mcp_server/
│   ├── clients/           # Database clients
│   │   ├── graphiti_client.py    # Knowledge graph client
│   │   └── rag_client.py         # Vector search client
│   ├── config.py          # Configuration management
│   ├── models.py          # Pydantic response models
│   ├── server.py          # FastMCP server & tools
│   └── __main__.py        # CLI entry point
├── tests/                 # Integration tests
├── cfg.yml                # Environment-based configuration
└── template-secrets.env   # Environment variables template
```

## Anti-Hallucination Design

This MCP server is specifically designed to prevent AI hallucination through:

1. **Fact Verification First**: `verify_fact` tool checks claims before making statements
2. **Source Attribution**: All responses clearly indicate source ([Graph] vs [Documents])  
3. **Graceful Degradation**: Returns "no information found" rather than guessing
4. **Clear Tool Guidance**: Each tool has explicit "USE THIS WHEN" instructions
5. **Structured Responses**: Pydantic models ensure consistent, validated output

The result: AI agents that provide accurate, grounded responses backed by verified knowledge.
