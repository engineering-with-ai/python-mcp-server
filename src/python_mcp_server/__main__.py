"""MCP server CLI entry point.

Two commands:

  python -m python_mcp_server         # start MCP stdio server (no seed)
  python -m python_mcp_server seed    # one-shot: run seed_all and exit

Seed is a SEPARATE process from the MCP child for a reason — pydantic-ai
spawns/kills the MCP child per request, so a background seed inside the
MCP child gets cancelled mid-bulk-load and the marker never writes
(infinite re-seed loop). Caller (docker compose / analyst-server image
entrypoint) runs `seed` first, THEN starts the long-lived analyst-server
process; once the marker is written, every subsequent MCP-child spawn
sees marker=True and skips seed cleanly.
"""

import asyncio
import sys

from .config import load_config, setup_logger
from .seed import seed_all
from .server import create_server


async def _serve() -> None:
    """Long-lived MCP stdio loop. No seed — caller should run `seed` first."""
    config = load_config()
    setup_logger(config)
    server = create_server(config)
    await server.run_stdio_async()


async def _seed() -> None:
    """One-shot seed — runs to completion, writes markers, exits."""
    config = load_config()
    setup_logger(config)
    await seed_all(
        vector_url=config.settings.vector_seed_url,
        graph_neo4j_url=config.settings.graph_neo4j_seed_url,
    )


def main() -> None:
    """Entry point for uvx/CLI execution."""
    if len(sys.argv) > 1 and sys.argv[1] == "seed":
        asyncio.run(_seed())
    else:
        asyncio.run(_serve())


if __name__ == "__main__":
    main()
