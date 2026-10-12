import os
import sys
from pathlib import Path

from agents import (
    Agent,
    ModelSettings,
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

instructions = """
You are a research planning assistant.

**TASK INSTRUCTIONS**
- You will be given a research topic and a list of allowed research sources.
- Constrain your research plan only to those sources.
- Output 5 concise tasks and specify which source is used for each task.
- Each task description must be five words or fewer.
"""


def research_source_names() -> list[str]:
    return [
        "Wikipedia",
        "Google",
        "Youtube",
    ]

################

@function_tool
def get_research_sources() -> list[str]:
    """Provides a list of research sources."""
    return research_source_names()


def build_groq_model() -> OpenAIResponsesModel:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print(
            "Missing GROQ_API_KEY. Add it to a .env file (see .env.example).",
            file=sys.stderr,
        )
        sys.exit(1)

    model_name = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
    client = AsyncOpenAI(base_url=GROQ_BASE_URL, api_key=api_key)
    set_default_openai_api("responses")
    set_tracing_disabled(True)
    return OpenAIResponsesModel(model=model_name, openai_client=client)


def main() -> None:
    agent = Agent(
        name="Research Planner",
        instructions=instructions,
        model=build_groq_model(),
        output_type=ResearchPlan,
        # Groq Responses API: structured output cannot be combined with tools.
        model_settings=ModelSettings(
            temperature=1.0,
            top_p=1.0,
            max_tokens=2048,
        ),
    )

    topic = "learn about AI agents"
    sources = research_source_names()
    user_input = (
        f"Research topic: {topic}\n"
        f"Available research sources: {', '.join(sources)}"
    )
    result = Runner.run_sync(agent, user_input)
    plan = result.final_output_as(ResearchPlan)
    for task in plan.tasks:
        print(f"{task.step}. [{task.research_source}] {task.description}")


if __name__ == "__main__":
    main()
