import os
import sys
from pathlib import Path

from agents import (
    Agent,
    ModelSettings,
    OpenAIChatCompletionsModel,
    OpenAIResponsesModel,
    Runner,
    function_tool,
    set_default_openai_api,
    set_tracing_disabled,
)
from dotenv import load_dotenv
from openai import AsyncOpenAI

from output_types import ResearchPlan

load_dotenv(Path(__file__).resolve().parent / ".env")

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"
# gpt-oss models often fail nested JSON schema on Groq; qwen is reliable for output_type.
DEFAULT_FORMAT_MODEL = "qwen/qwen3.8-27b"

RESEARCH_INSTRUCTIONS = """
You are a research planning assistant.

**TASK INSTRUCTIONS**
- You will be given a research topic.
- Begin by calling get_research_sources() to learn which sources are allowed.
- Propose exactly 5 research tasks for the topic using only those sources.
- For each task, state the step number, which source to use, and a short description (five words or fewer).
- Write a clear draft plan in plain text; a downstream agent will structure it.
"""

FORMAT_INSTRUCTIONS = """
You receive a draft research plan from another agent.

Convert it into the required structured output.
- Preserve exactly 5 tasks with step (1-5), research_source, and description.
- research_source must be exactly one of: Wikipedia, Google, Youtube.
- Each description must be five words or fewer (strict).
"""

MODEL_SETTINGS = ModelSettings(
    temperature=1.0,
    top_p=1.0,
    max_tokens=2048,
)


@function_tool
def get_research_sources() -> list[str]:
    """Provides a list of research sources."""
    return [
        "Wikipedia",
        "Google",
        "Youtube",
    ]


def _require_groq_api_key() -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print(
            "Missing GROQ_API_KEY. Add it to a .env file (see .env.example).",
            file=sys.stderr,
        )
        sys.exit(1)
    return api_key


def build_groq_model(model_name: str | None = None) -> OpenAIResponsesModel:
    if model_name is None:
        model_name = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
    client = AsyncOpenAI(base_url=GROQ_BASE_URL, api_key=_require_groq_api_key())
    set_default_openai_api("responses")
    set_tracing_disabled(True)
    return OpenAIResponsesModel(model=model_name, openai_client=client)


def build_groq_chat_model(model_name: str | None = None) -> OpenAIChatCompletionsModel:
    """Chat Completions on Groq (required for MCP stdio tools; Responses returns 400)."""
    if model_name is None:
        model_name = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
    client = AsyncOpenAI(base_url=GROQ_BASE_URL, api_key=_require_groq_api_key())
    set_default_openai_api("chat_completions")
    set_tracing_disabled(True)
    return OpenAIChatCompletionsModel(model=model_name, openai_client=client)


def main() -> None:
    research_model = build_groq_model()
    format_model_name = os.getenv("GROQ_FORMAT_MODEL", DEFAULT_FORMAT_MODEL)
    format_model = build_groq_model(format_model_name)

    research_agent = Agent(
        name="Research Agent",
        instructions=RESEARCH_INSTRUCTIONS,
        model=research_model,
        tools=[get_research_sources],
        model_settings=MODEL_SETTINGS,
    )

    format_agent = Agent(
        name="Format Agent",
        instructions=FORMAT_INSTRUCTIONS,
        model=format_model,
        output_type=ResearchPlan,
        model_settings=ModelSettings(
            temperature=0.3,
            top_p=1.0,
            max_tokens=2048,
        ),
    )

    topic = "learn about AI agents"
    research_result = Runner.run_sync(
        research_agent,
        f"Research topic: {topic}",
    )

    format_result = Runner.run_sync(
        format_agent,
        f"Research topic: {topic}\n\nDraft plan:\n{research_result.final_output}",
    )

    plan = format_result.final_output_as(ResearchPlan)
    for task in plan.tasks:
        print(f"{task.step}. [{task.research_source}] {task.description}")


if __name__ == "__main__":
    main()
