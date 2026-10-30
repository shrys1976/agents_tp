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


class ResearchOutputModel(BaseModel):
    """Output model for the research agent."""

    model_config = ConfigDict(extra="forbid")

    research_plan: str = Field(..., description="The final research plan as text.")
    research_plan_file: str = Field(
        ...,
        description="The path to the research plan file.",
    )
    is_sufficiently_detailed: bool = Field(
        ...,
        description=(
            "True if the plan is at least 100 characters and includes "
            "actionable research steps."
        ),
    )


class ResearchInputModel(BaseModel):
    """Structured result from the input guardrail agent."""

    model_config = ConfigDict(extra="forbid")

    research_validation: str = Field(
        ...,
        description="Short explanation of the validation result.",
    )
    is_research_forbidden: bool = Field(
        ...,
        description="True if the request targets a forbidden research topic.",
    )


class ResearchOutputValidationModel(BaseModel):
    """Structured result from the output guardrail agent."""

    model_config = ConfigDict(extra="forbid")

    output_validation: str = Field(
        ...,
        description="Short explanation of the output validation result.",
    )
    is_output_insufficient: bool = Field(
        ...,
        description="True if the research plan fails quality checks.",
    )
