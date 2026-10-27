# research mcp server for agent
import asyncio
from pathlib import Path

from agents import Agent, ModelSettings, Runner
from agents.mcp import MCPServerSse

from mcp_stdio import sse_server_process
from output_types import ResearchPlan
from research_agent import(

    FORMAT_INSTRUCTIONS,
    MODEL_SETTINGS,
    build_groq_chat_model,
    build_groq_model,
    DEFAULT_FORMAT_MODEL,
)

import os


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


RESEARCH_MCP_INSTRUCTIONS = """
You are a research planning assistant.
**TASK INSTRUCTIONS**
- You will be given a research topic.
- Begin by calling the MCP tool get_research_sources to learn which sources are allowed.
- Propose exactly 5 research tasks using only those sources.
- For each task: step number, source, short description (five words or fewer).
- Write a clear plain-text draft; a downstream agent will structure it.
"""


async def main() -> None:
    topic = "learn about AI agents"

    async with sse_server_process():
        async with MCPServerSse(

            name="Research Tools",
            params=SSE_PARAMS,
            cache_tools_list=True,
            client_session_timeout_seconds=60,
        )as mcp_server:

            research_agent =  Agent(
                name="Research Agent",
                instructions=RESEARCH_MCP_INSTRUCTIONS,
                model=build_groq_chat_model(),
                mcp_server=[mcp_server],
                model_settings=MODEL_SETTINGS,
            )

            format_agent = Agent(
                name ="Format Agent",
                instructions=FORMAT_INSTRUCTIONS,
                model=build_groq_model(
                    os.getenv("GROQ_FORMAT_MODEL",DEFAULT_FORMAT_MODEL)
                ),
                output_type = ResearchPlan,
                model_settings=ModelSettings(
                    temperature = 0.3,
                    top_p = 1.0,
                    max_tokens = 2048,
                ),

            )
            research_result =  await Runner.run(
                research_agent,
                f"Research topic" :{topic}",

            )

            format_result = await Runner.run(
                format_agent,
                f"Research topic: {topic}\n\nDraft plan:\n{research_result.final_output}",
            )

            
            plan = format_result.final_output_as(ResearchPlan)
            for task in plan.tasks:
                print(f"{task.step}. [{task.research_source}] {task.description}")
if __name__ == "__main__":
    asyncio.run(main())
