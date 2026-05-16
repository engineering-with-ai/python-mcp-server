"""MCP server CLI entry point."""

import asyncio

from .config import load_config, setup_logger
from .seed import seed_all
from .server import create_server


async def _run() -> None:
    config = load_config()
    setup_logger(config)
    await seed_all(
        vector_url=config.settings.vector_seed_url,
        graph_neo4j_url=config.settings.graph_neo4j_seed_url,
    )
    server = create_server(config)
    await server.run_stdio_async()


def main() -> None:
    """Entry point for uvx/CLI execution."""
    asyncio.run(_run())


if __name__ == "__main__":
    main()
