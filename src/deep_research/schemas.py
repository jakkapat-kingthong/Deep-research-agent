"""Core pydantic schemas — the contract for every LLM boundary."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class SubQuestion(BaseModel):
    """A decomposed sub-question from the planner."""

    text: str = Field(min_length=3, max_length=300)
    rationale: str = Field(default="", max_length=500)


class Source(BaseModel):
    """A source (article) in the research state."""

    source_id: str = Field(pattern=r"^src_\d+$")  # e.g. src_01, src_02
    url: str
    title: str = ""
    author: str = ""
    date: str = ""


class Claim(BaseModel):
    """A single claim in the final report. MUST have source grounding."""

    text: str = Field(min_length=5, max_length=1000)
    source_ids: list[str] = Field(min_length=1, description="Must be non-empty")
    confidence: float = Field(ge=0.0, le=1.0, default=0.8)

    @field_validator("source_ids")
    @classmethod
    def validate_source_ids_format(cls, v: list[str]) -> list[str]:
        for sid in v:
            if not sid.startswith("src_"):
                raise ValueError(f"source_id must start with 'src_', got {sid!r}")
        return v


class Report(BaseModel):
    """Final research report."""

    query: str
    summary: str = Field(min_length=20)
    claims: list[Claim] = Field(min_length=1)
    sources: list[Source] = Field(min_length=1)

    def validate_grounding(self) -> list[str]:
        """Return list of source_ids referenced by claims but missing from sources."""
        source_ids_avail = {s.source_id for s in self.sources}
        missing: set[str] = set()
        for claim in self.claims:
            for sid in claim.source_ids:
                if sid not in source_ids_avail:
                    missing.add(sid)
        return sorted(missing)


class PlannerOutput(BaseModel):
    """Structured output from the planner node."""

    sub_questions: list[SubQuestion] = Field(min_length=1, max_length=5)
    reasoning: str = Field(default="")
