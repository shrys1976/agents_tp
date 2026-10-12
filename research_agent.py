import os
import sys
from pathlib import Path

from agents import (
    Agent,
    ModelSettings,
    OpenAIResponsesModel,
    Runner,
    set_default_openai_api,
    set_tracing_disabled,
)
from dotenv import load_dotenv
from openai import AsyncOpenAI

from output_types import ResearchPlan

load_dotenv(Path(__file__).resolve().parent / ".env")

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
# Groq Responses API (beta); override via GROQ_MODEL in .env.
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"

instructions = """
You are a research planning assistant.

**TASK INSTRUCTIONS**
- You will be given a research topic
- Your task is to provide a plan on how to research this topic.
- Output 5 concise tasks (5 words or less) to your plan.

"""


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
    # Agents SDK defaults to Responses; make that explicit for Groq's /v1/responses endpoint.
    set_default_openai_api("responses")
    set_tracing_disabled(True)
    return OpenAIResponsesModel(model=model_name, openai_client=client)


def main() -> None:
    agent = Agent(
        name="Research Planner",
        instructions=instructions,
        model=build_groq_model(),
        output_type=ResearchPlan,
        model_settings=ModelSettings(
            temperature=1.0,
            top_p=1.0,
            # Reasoning models need headroom; 150 triggers response.incomplete (max_output_tokens).
            max_tokens=1024,
        ),
    )

    user_input = "learn about AI agents"
    result = Runner.run_sync(agent, user_input)
    plan = result.final_output_as(ResearchPlan)
    for index, task in enumerate(plan.tasks, start=1):
        print(f"{index}. {task}")


if __name__ == "__main__":
    main()
