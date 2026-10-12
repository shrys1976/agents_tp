from pydantic import BaseModel, Field


class ResearchPlan(BaseModel):
    """Research plan: exactly five short tasks for a given topic."""

    tasks: list[str] = Field(
        ...,
        min_length=5,
        max_length=5,
        description="Five research tasks; each task should be five words or fewer.",
    )
