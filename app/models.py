"""Data models and the provider/model catalog.

The structured-output schemas mirror the Pydantic models used in the original
notebook (``ResumeOutput`` and ``CoverLetterOutput``) and add a structured
schema for the resume-vs-job gap analysis so each section can be rendered
separately in the UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

Provider = Literal["openai", "anthropic"]


@dataclass(frozen=True)
class ProviderConfig:
    """Static configuration for one LLM provider."""

    label: str
    env_var: str
    models: tuple[str, ...]
    key_url: str

    @property
    def default_model(self) -> str:
        return self.models[0]


# The first model in each tuple is the default. ``gpt-4o`` and
# ``claude-sonnet-4-6`` are the models used in the original notebook.
PROVIDERS: dict[Provider, ProviderConfig] = {
    "openai": ProviderConfig(
        label="OpenAI",
        env_var="OPENAI_API_KEY",
        models=("gpt-4o", "gpt-4o-mini", "gpt-4.1"),
        key_url="https://platform.openai.com/api-keys",
    ),
    "anthropic": ProviderConfig(
        label="Anthropic Claude",
        env_var="ANTHROPIC_API_KEY",
        models=("claude-sonnet-4-6", "claude-opus-5-5", "claude-haiku-4-5"),
        key_url="https://console.anthropic.com/settings/keys",
    ),
}


class OverallMatch(BaseModel):
    """Overall alignment between the resume and the job description."""

    match_level: Literal["Strong", "Moderate", "Weak"] = Field(
        description="Overall fit, judged only from evidence in the resume."
    )
    summary: str = Field(description="A short, evidence-based explanation of the overall alignment.")
    priority_actions: list[str] = Field(description="The most important, honest improvements the candidate could make.")


class GapAnalysis(BaseModel):
    """Structured comparison of a resume against a job description."""

    key_requirements: list[str] = Field(
        description="Main skills, qualifications, technologies, experience and responsibilities sought."
    )
    relevant_experience: list[str] = Field(
        description="Resume content that matches or closely aligns with the requirements."
    )
    gaps_and_mismatches: list[str] = Field(
        description="Requirements that are missing, unclear or underrepresented in the resume."
    )
    potential_strengths: list[str] = Field(
        description="Aligned strengths plus valuable extras not explicitly requested."
    )
    overall_match: OverallMatch


class TailoredResume(BaseModel):
    """A resume rewritten for a target job (``ResumeOutput`` in the notebook)."""

    updated_resume: str = Field(description="The full rewritten resume as Markdown text.")
    change_notes: list[str] = Field(description="Brief notes explaining the main changes and why they were made.")


class CoverLetter(BaseModel):
    """A cover letter for the target job (``CoverLetterOutput`` in the notebook)."""

    cover_letter: str
