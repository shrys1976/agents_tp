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

from output_types import (
    ResearchInputModel,
    ResearchOutputModel,
    ResearchOutputValidationModel,
)
from research_agent import DEFAULT_FORMAT_MODEL, build_groq_chat_model
from single_agent_multiple_mcp import servers

load_dotenv(Path(__file__).resolve().parent / ".env")

FORBIDDEN_RESEARCH = "The Hitchhiker's Guide to the Galaxy"

_guardrail_model = build_groq_chat_model()
_chat_model = build_groq_chat_model()
_format_model = build_groq_chat_model(
    os.getenv("GROQ_FORMAT_MODEL", DEFAULT_FORMAT_MODEL),
)


def _input_as_text(input: str | list[TResponseInputItem]) -> str:
    if isinstance(input, str):
        return input
    return str(input)


input_guardrail_agent = Agent(
    name="Input guardrail agent",
    instructions=f"""
You are an input guardrail agent.
Reject research requests about this forbidden topic:
"{FORBIDDEN_RESEARCH}"

Return structured output:
- research_validation: brief explanation of your decision
- is_research_forbidden: true only if the request is about the forbidden topic
""",
    model=_guardrail_model,
    output_type=ResearchInputModel,
)

output_guardrail_agent = Agent(
    name="Output guardrail agent",
    instructions="""
You are an output guardrail agent.
Judge whether the research plan is sufficient.

Treat the plan as insufficient if any of these apply:
- The plan text is shorter than 100 characters
- The plan lacks actionable research steps
- is_sufficiently_detailed is false

Return structured output:
- output_validation: brief explanation of your decision
- is_output_insufficient: true if the plan fails any quality check
""",
    model=_guardrail_model,
    output_type=ResearchOutputValidationModel,
)


@input_guardrail
async def research_guardrail(
    ctx: RunContextWrapper[None],
    agent: Agent,
    input: str | list[TResponseInputItem],
) -> GuardrailFunctionOutput:
    prompt = f"Validate this research request:\n{_input_as_text(input)}"
    result = await Runner.run(
        input_guardrail_agent,
        prompt,
        context=ctx.context,
    )
    validation = result.final_output_as(ResearchInputModel)
    return GuardrailFunctionOutput(
        output_info=validation.research_validation,
        tripwire_triggered=validation.is_research_forbidden,
    )


@output_guardrail
async def research_output_guardrail(
    ctx: RunContextWrapper,
    agent: Agent,
    output: ResearchOutputModel,
) -> GuardrailFunctionOutput:
    prompt = f"""Validate this research output:

Research plan:
{output.research_plan}

File path: {output.research_plan_file}
Claimed sufficiently detailed: {output.is_sufficiently_detailed}
"""
    result = await Runner.run(
        output_guardrail_agent,
        prompt,
        context=ctx.context,
    )
    validation = result.final_output_as(ResearchOutputValidationModel)
    return GuardrailFunctionOutput(
        output_info=validation.output_validation,
        tripwire_triggered=validation.is_output_insufficient,
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
    Return structured output with:
    - research_plan: the full plan text
    - research_plan_file: path to the file you wrote
    - is_sufficiently_detailed: true if the plan is at least 100 characters and
      includes actionable research steps
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
