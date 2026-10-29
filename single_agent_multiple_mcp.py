
# mcp client (single agent calling tools from 3 different mcp servers over stdio)
import asyncio
from pathlib import Path

from agents import Agent, Runner
from agents.mcp import MCPServerStdio, MCPServerStdioParams

SCRIPT = Path(__file__).resolve().parent / "research_mcp_server.py"
SANDBOX = str(Path(__file__).resolve().parent)

# 3 mcp servers for use for a single agent
servers = [
    MCPServerStdio(
        name="Research Tools",
        params=MCPServerStdioParams(
            command="mcp",
            args=["run", str(SCRIPT)],
        ),
    ),

    MCPServerStdio(
        name="sequential-thinking",
        params={
            "command": "npx",
            "args" :[
                "-y",
                "@modelcontextprotocol/server-sequential-thinking"
            ],
        },
    ),

    MCPServerStdio(
        name="filesystem",
        params=MCPServerStdioParams(
            command="npx",
            args=[
                "-y",
                "@modelcontextprotocol/server-filesystem", SANDBOX
            ],
        ),
    ),

]

instructions = """
You are a research assistant who can use tools to perform and plan research.
Given a research goal, use the research tools to find research sources.
Then, use the sequential thinking tool to plan the research.
Finally, use the filesystem tool to write the research plan as a text file.
"""


async def main() -> None:
    async with (
        servers[0] as research_srv,
        servers[1] as thinking_srv,
        servers[2] as fs_srv,
    ):

        agent = Agent(
            name="Research Assistant",
            instructions = instructions,
            mcp_servers=[research_srv,thinking_srv,fs_srv],
        )

        goal= """
    Produce a research plan to find the book: 'Mahabharata by Bibek Debroy'
    """

        print("Running...", goal)
        result = await Runner.run(agent,goal)
        print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
