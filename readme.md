# Python MCP Server 🧠

![](https://img.shields.io/gitlab/pipeline-status/engineering-with-ai/python-mcp-server?branch=main&logo=gitlab)
![](https://gitlab.com/engineering-with-ai/python-mcp-server/badges/main/coverage.svg)
![](https://img.shields.io/badge/3.13.2-gray?logo=python)
![](https://img.shields.io/badge/0.10.9-gray?logo=uv)
![](https://img.shields.io/badge/5.0.0-gray?logo=neo4j)
![](https://img.shields.io/badge/16.0.0-white?logo=postgresql)

A Model Context Protocol (MCP) server that gives AI agents access to a Graphiti
knowledge graph and a pgvector document store for grounded, evidence-backed
responses.

## Features

- **🔍 Hybrid graph search** — semantic + BM25 + graph traversal via Graphiti
- **📚 Vector RAG** — pgvector similarity search; query strings are embedded
  internally via OpenAI (no pre-computed vectors required from callers)
- **🧾 Evidence retrieval, not fake verification** — `verify_fact` returns
  related graph evidence; the calling LLM judges entailment
- **⚡ Fail fast** — client errors surface as MCP errors, not silent empty results
- **⚙️ Clean config split** — `cfg.yml` for config, env vars for secrets only

## Tools

| Tool              | Input                       | Returns                                |
| ----------------- | --------------------------- | -------------------------------------- |
| `search_knowledge`| `query: str`                | Graph entities/relationships           |
| `rag_search`      | `query: str`                | Document chunks ranked by similarity   |
| `verify_fact`     | `statement: str`            | `FactEvidence { statement, evidence }` |
| `combined_search` | `query: str`                | Graph results + document chunks        |

All tools take strings — embeddings are generated server-side.

Resources: `knowledge://instructions`, `knowledge://examples`.
Prompt: `answer_with_verification`.

## Quick Start

```bash
pip install python-mcp-server
# or
uvx python-mcp-server
```

## Configuration

`cfg.yml` holds all non-secret config. Secrets live only in environment
variables — never in `cfg.yml`, never in the Postgres URL.

### `cfg.yml`

```yaml
local:
  log_level: DEBUG
  neo4j:
    uri: bolt://localhost:7687
    user: neo4j
    database: neo4j
  postgres:
    host: localhost
    port: 5432
    database: knowledge
    user: postgres
    embeddings_table: energy_embeddings
    embedding_model: text-embedding-3-small
```

The server selects the top-level key based on the `ENV` env var (default `local`).
A `beta` section is also supported.

### Secrets (environment)

```bash
export NEO4J_PASSWORD="..."
export POSTGRES_PASSWORD="..."
export OPENAI_API_KEY="..."
export ENV="local"
```

### Programmatic usage

```python
from python_mcp_server import create_server
from python_mcp_server.config import Config, Neo4jConfig, PostgresConfig, LogLevel

config = Config(
    log_level=LogLevel.INFO,
    neo4j=Neo4jConfig(uri="bolt://localhost:7687", user="neo4j", database="neo4j"),
    postgres=PostgresConfig(
        host="localhost", port=5432, database="knowledge", user="postgres",
        embeddings_table="energy_embeddings",
        embedding_model="text-embedding-3-small",
    ),
)
server = create_server(
    config=config,
    neo4j_password="...",
    postgres_password="...",
    openai_api_key="...",
)
```

## Usage

### Claude Code

```bash
export NEO4J_PASSWORD=... POSTGRES_PASSWORD=... OPENAI_API_KEY=...
claude mcp add domain-expert -- uvx python-mcp-server
```

### Claude Desktop (`claude_desktop_config.json`)

```json
{
  "mcpServers": {
    "knowledge-graph": {
      "command": "uvx",
      "args": ["python-mcp-server"],
      "env": {
        "NEO4J_PASSWORD": "...",
        "POSTGRES_PASSWORD": "...",
        "OPENAI_API_KEY": "..."
      }
    }
  }
}
```

### Pydantic-AI

```python
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio

mcp = MCPServerStdio("uvx", "python-mcp-server")
agent = Agent(toolsets=[mcp])
result = await agent.run("What connects Tesla and battery technology?")
```

## Database Schema

pgvector table expected by `rag_search`:

```sql
CREATE TABLE energy_embeddings (
    id SERIAL PRIMARY KEY,
    title TEXT,
    content TEXT NOT NULL,
    book TEXT,
    section_level TEXT,
    analysis_relevance TEXT,
    embedding vector(1536),  -- text-embedding-3-small
    content_tsv tsvector
        GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
);

CREATE INDEX ON energy_embeddings USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX idx_content_tsv ON energy_embeddings USING gin(content_tsv);
```

Embedding dimension must match `embedding_model` in `cfg.yml`.

`rag_search` issues two rankings against this table — cosine over `embedding`
and BM25 over `content_tsv` — and fuses them via Reciprocal Rank Fusion
(k=60). Exact-term matches (protocol field names, enum values, requirement
IDs) come through the BM25 leg that pure cosine would miss.

## Development

```bash
git clone <repo> && cd python-mcp-server
uv sync --dev
cp template-secrets.env .env  # fill in secrets

uv run poe checks   # deptry, black, ruff, mypy, bandit, pip-audit
uv run poe cover    # tests with coverage
uv run python-mcp-server
```

## Architecture

```
src/python_mcp_server/
├── clients/
│   ├── embedder.py         # OpenAI embeddings (injected)
│   ├── graphiti_client.py  # Neo4j via Graphiti
│   └── rag_client.py       # pgvector similarity search
├── config.py               # cfg.yml loader
├── models.py               # Pydantic response models
├── server.py               # FastMCP tools, resources, prompt
└── __main__.py             # CLI entry point
```

## Design Principles

1. **String-in, evidence-out.** Callers pass natural language; the server
   handles embeddings and returns typed Pydantic results.
2. **No fake verification.** `verify_fact` returns evidence; the caller LLM
   decides entailment. The server never invents a `verified: bool`.
3. **Fail fast.** Database errors propagate to the MCP client so Claude sees
   "Neo4j unreachable" instead of "no results."
4. **Config vs. secrets are separate concerns.** `cfg.yml` is checked in;
   passwords and API keys never are.
