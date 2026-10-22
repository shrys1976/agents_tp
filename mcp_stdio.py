"""Run an agent against the FastMCP server in research_mcp_server.py over SSE."""

import asyncio
import contextlib
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

from agents import Agent, Runner
from agents.mcp import MCPServerSse, MCPServerSseParams
from dotenv import load_dotenv

from research_agent import build_groq_chat_model

load_dotenv(Path(__file__).resolve().parent / ".env")

SCRIPT = Path(__file__).resolve().parent / "research_mcp_server.py"
PROJECT_DIR = SCRIPT.parent

MCP_HOST = os.getenv("MCP_SSE_HOST", "127.0.0.1")
MCP_PORT = int(os.getenv("MCP_SSE_PORT", "8000"))
SSE_URL = os.getenv("MCP_SSE_URL", f"http://{MCP_HOST}:{MCP_PORT}/sse")

SSE_PARAMS: MCPServerSseParams = {
    "url": SSE_URL,
    "timeout": 30,
    "sse_read_timeout": 300,
}


def _sse_server_is_up(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError):
        return False


async def wait_for_sse_server(url: str, *, timeout_seconds: float = 30) -> None:
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    while asyncio.get_running_loop().time() < deadline:
        if await asyncio.to_thread(_sse_server_is_up, url):
            return
        await asyncio.sleep(0.25)
    print(f"SSE server did not become ready at {url}", file=sys.stderr)
    sys.exit(1)


@contextlib.asynccontextmanager
async def sse_server_process():
    process = await asyncio.create_subprocess_exec(
        "uv",
        "run",
        "fastmcp",
        "run",
        str(SCRIPT),
        "-t",
        "sse",
        "--host",
        MCP_HOST,
        "-p",
        str(MCP_PORT),
        "--no-banner",
        cwd=str(PROJECT_DIR),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        await wait_for_sse_server(SSE_URL)
        yield process
    finally:
        process.terminate()
        with contextlib.suppress(ProcessLookupError):
            await asyncio.wait_for(process.wait(), timeout=5)


async def main() -> None:
    async with sse_server_process():
        async with MCPServerSse(
            name="Research Tools",
            params=SSE_PARAMS,
            cache_tools_list=True,
            client_session_timeout_seconds=60,
        ) as research_server:
            agent = Agent(
                name="Assistant",
                instructions=(
                    "Use the research MCP tools when needed. "
                    "Call get_research_sources to list allowed sources."
                ),
                model=build_groq_chat_model(),
                mcp_servers=[research_server],
            )

            print(f"Running against SSE server at {SSE_URL}")
            result = await Runner.run(
                agent,
                "Get the available research sources",
            )
            print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
