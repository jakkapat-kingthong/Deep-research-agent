"""Tests for pydantic schema validators."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from deep_research.schemas import Claim, Report, Source


def test_claim_requires_source_ids() -> None:
    with pytest.raises(ValidationError):
        Claim(text="some claim", source_ids=[])


def test_claim_rejects_bad_source_id_format() -> None:
    with pytest.raises(ValidationError):
        Claim(text="some claim", source_ids=["bad_format"])


def test_report_validates_grounding() -> None:
    sources = [Source(source_id="src_01", url="https://a.com")]
    claims = [Claim(text="x is y", source_ids=["src_01", "src_99"])]
    report = Report(
        query="q", summary="this is a much longer summary here", claims=claims, sources=sources
    )
    missing = report.validate_grounding()
    assert missing == ["src_99"]
