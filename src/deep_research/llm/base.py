"""LLM Protocol — shared interface for Claude + Gemini."""

from __future__ import annotations

from typing import Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMProvider(Protocol):
    """Shared interface for all LLM providers."""

    async def structured_complete(
        self,
        *,
        system: str,
        user: str,
        response_model: type[T],
        max_tokens: int = 2048,
    ) -> tuple[T, int, int]:
        """Return (parsed_model, tokens_in, tokens_out)."""
