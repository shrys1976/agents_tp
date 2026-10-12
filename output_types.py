from pydantic import BaseModel, ConfigDict, Field


class ResearchTask(BaseModel):
    """One step in a research plan tied to a allowed source."""

    model_config = ConfigDict(extra="forbid")

    step: int = Field(..., ge=1, le=5, description="Task number from 1 to 5.")
    research_source: str = Field(
        ...,
        description="One of the sources returned by get_research_sources.",
    )
    description: str = Field(
        ...,
        description="What to do; five words or fewer.",
    )


class ResearchPlan(BaseModel):
    """Structured research plan for a topic."""

    model_config = ConfigDict(extra="forbid")

    tasks: list[ResearchTask] = Field(
        ...,
        min_length=5,
        max_length=5,
        description="Five numbered tasks using only allowed research sources.",
    )
