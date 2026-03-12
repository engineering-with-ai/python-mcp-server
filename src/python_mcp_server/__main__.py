"""MCP server CLI entry point."""

import asyncio

from .server import create_server


def main() -> None:
    """Entry point for uvx/CLI execution."""
    server = create_server()
    asyncio.run(server.run_stdio_async())


if __name__ == "__main__":
    main()
