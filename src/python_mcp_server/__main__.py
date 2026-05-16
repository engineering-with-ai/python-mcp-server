"""MCP server CLI entry point."""

import asyncio
import logging

from .config import Config, load_config, setup_logger
from .seed import seed_all
from .server import create_server


async def _seed_in_background(config: Config) -> None:
    """Run seed_all without blocking MCP boot. Logs on failure.

    Why background: Neptune Bulk Loader takes 3-5 min for the production
    CSVs. Blocking MCP initialize on seed means the first /chat after a
    fresh deploy hangs for that whole window (pydantic-ai stdio client
    times out at 30s default). Backgrounding lets MCP respond to the
    initialize handshake immediately; tool calls during seed return empty
    results, which is a graceful degradation vs a 500.

    Errors are LOGGED but not raised — asyncio background-task exceptions
    silently disappear without a handler. The marker pattern means a
    failed seed leaves no marker, so the NEXT boot retries.
    """
    try:
        await seed_all(
            vector_url=config.settings.vector_seed_url,
            graph_neo4j_url=config.settings.graph_neo4j_seed_url,
        )
    except Exception:
        logging.getLogger(__name__).exception("background seed failed")


async def _run() -> None:
    config = load_config()
    setup_logger(config)
    # Kick seed off as a background task so MCP stdio becomes responsive
    # immediately. See _seed_in_background for the why.
    asyncio.create_task(_seed_in_background(config))
    server = create_server(config)
    await server.run_stdio_async()


def main() -> None:
    """Entry point for uvx/CLI execution."""
    asyncio.run(_run())


if __name__ == "__main__":
    main()
