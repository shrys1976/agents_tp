import asyncio
import os
from pathlib import Path

from agents import (
    Agent,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    OutputGuardrailTripwireTriggered,
    RunContextWrapper,
    Runner,
    input_guardrail,
    output_guardrail,
)
from agents.items import TResponseInputItem
from dotenv import load_dotenv

from output_types import ResearchOutputModel
from research_agent import DEFAULT_FORMAT_MODEL, build_groq_chat_model
from single_agent_multiple_mcp import servers

load_dotenv(Path(__file__).resolve().parent / ".env")

FORBIDDEN_RESEARCH = "The Hitchhiker's Guide to the Galaxy"


def _input_as_text(input: str | list[TResponseInputItem]) -> str:
    if isinstance(input, str):
        return input
    return str(input)


@input_guardrail
async def research_guardrail(
    ctx: RunContextWrapper[None],
    agent: Agent,
    input: str | list[TResponseInputItem],
) -> GuardrailFunctionOutput:
    text = _input_as_text(input)
    research_forbidden = FORBIDDEN_RESEARCH in text
    return GuardrailFunctionOutput(
        output_info=f"user asked: {text}",
        tripwire_triggered=research_forbidden,
    )


@output_guardrail
async def research_output_guardrail(
    ctx: RunContextWrapper,
    agent: Agent,
    output: ResearchOutputModel,
) -> GuardrailFunctionOutput:
    insufficient_research = len(output.research_plan) < 100
    return GuardrailFunctionOutput(
        output_info=f"research plan length: {len(output.research_plan)}",
        tripwire_triggered=insufficient_research,
    )


_chat_model = build_groq_chat_model()
_format_model = build_groq_chat_model(
    os.getenv("GROQ_FORMAT_MODEL", DEFAULT_FORMAT_MODEL),
)

research_agent = Agent(
    name="Research Agent",
    instructions="""
    You are a research assistant.
    Your role is to find research sources.
    """,
    model=_chat_model,
    input_guardrails=[research_guardrail],
)

thinking_agent = Agent(
    name="Thinking Agent",
    instructions="""
    You are a research assistant.
    Your role is to plan the research.
    """,
    model=_chat_model,
)

filesystem_agent = Agent(
    name="Filsystem Agent",
    instructions="""
    You are a research assistant.
    Your role is to write the research plan as a text file using the filesystem tools.
    Return structured output with the full research plan text and the path to the file
    you wrote.
    """,
    model=_format_model,
    output_type=ResearchOutputModel,
    output_guardrails=[research_output_guardrail],
)


async def main() -> None:
    async with (
        servers[0] as research_srv,
        servers[1] as thinking_srv,
        servers[2] as fs_srv,
    ):
        goal = """
    Produce a research plan to find the book: 'Mahabharata by Bibek Debroy'
    """

        try:
            print("Running...", goal)
            research_agent.mcp_servers = [research_srv]
            result = await Runner.run(research_agent, goal)
            thinking_agent.mcp_servers = [thinking_srv]
            result = await Runner.run(thinking_agent, result.final_output)
            filesystem_agent.mcp_servers = [fs_srv]
            result = await Runner.run(filesystem_agent, result.final_output)
            print(result.final_output)
        except InputGuardrailTripwireTriggered as input_tripped:
            print(f"""
Input guardrail tripwire triggered:
{input_tripped.guardrail_result.output.output_info}
""")
        except OutputGuardrailTripwireTriggered as output_tripped:
            print(f"""
Output guardrail tripwire triggered:
{output_tripped.guardrail_result.output.output_info}
""")


if __name__ == "__main__":
    asyncio.run(main())
