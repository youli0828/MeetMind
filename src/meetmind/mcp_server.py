"""MCP adapters for the same tools used by the Web workspace and agent."""

import argparse
import json
import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from openai import OpenAIError

from meetmind.api import get_service
from meetmind.errors import MeetMindError


def create_server(*, service_factory=get_service, host="127.0.0.1", port=8001) -> FastMCP:
    server = FastMCP("MeetMind", host=host, port=port)

    def call(name, arguments):
        try:
            return service_factory().tools.call(name, json.dumps(arguments))
        except (MeetMindError, OpenAIError, ValueError):
            raise ValueError(
                "Meeting tool failed; check arguments and model configuration"
            ) from None

    @server.tool()
    def search_meetings(query: str, k: int = 5) -> dict:
        """Hybrid search; return source snippets, meeting IDs and start/end seconds (k=1..10)."""
        return call("search_meetings", {"query": query, "k": k})

    @server.tool()
    def get_meeting_summary(meeting_id: str) -> dict:
        """Read stored meeting minutes, including summary, decisions and indexed action items."""
        return call("get_meeting", {"meeting_id": meeting_id})["minutes"]

    @server.tool()
    def create_todo(meeting_id: str, action_index: int) -> dict:
        """Save an existing action item locally; action_index is zero-based and idempotent."""
        return call("create_todo", {"meeting_id": meeting_id, "action_index": action_index})

    @server.tool()
    def create_calendar_event(meeting_id: str, title: str, start: str, end: str) -> dict:
        """Generate local ICS; require confirmed ISO 8601 times with timezone offsets."""
        return call(
            "create_calendar_event",
            {"meeting_id": meeting_id, "title": title, "start": start, "end": end},
        )

    @server.tool()
    def export_markdown(meeting_id: str) -> dict:
        """Export minutes and timestamped transcript; return an artifact download URL."""
        return call("export_markdown", {"meeting_id": meeting_id})

    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MeetMind MCP server")
    parser.add_argument("--transport", choices=("stdio", "sse"), default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument(
        "--directory", type=Path, help="Working directory for existing .env and data"
    )
    args = parser.parse_args(argv)
    if args.directory is not None:
        os.chdir(args.directory)
    create_server(host=args.host, port=args.port).run(transport=args.transport)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
