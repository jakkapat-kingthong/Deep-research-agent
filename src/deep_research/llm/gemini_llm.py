"""Google Gemini wrapper with structured output support."""

from __future__ import annotations

from typing import TypeVar

from google import genai
from google.genai import types
from loguru import logger
from pydantic import BaseModel

from deep_research.config import settings

T = TypeVar("T", bound=BaseModel)


class GeminiLLM:
    """Gemini wrapper. Enforces JSON schema compliance via response_schema."""

    def __init__(self, model: str = settings.PLANNER_MODEL) -> None:
        self._model = model
        self._client = genai.Client(api_key=settings.GOOGLE_API_KEY.get_secret_value())

    async def structured_complete(
        self,
        *,
        system: str,
        user: str,
        response_model: type[T],
        max_tokens: int = 2048,
    ) -> tuple[T, int, int]:
        """Force JSON output matching response_model."""

        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.2,  # Low temperature for reasoning tasks
            max_output_tokens=max_tokens,
            response_mime_type="application/json",
            response_schema=response_model,
        )

        # We use standard generate_content (blocking) since GenAI SDK's async support
        # can sometimes be tricky. LangGraph handles the threading for us.
        resp = self._client.models.generate_content(
            model=self._model,
            contents=user,
            config=config,
        )

        # Check if blocked by safety settings
        if not resp.text:
            logger.error(f"Gemini response empty. Candidates: {resp.candidates}")
            raise ValueError("Gemini returned empty text, likely blocked by safety settings.")

        # Ensure we parse the string output back into the Pydantic model
        parsed = response_model.model_validate_json(resp.text)

        # Gemini Usage Metadata
        tokens_in = resp.usage_metadata.prompt_token_count if resp.usage_metadata else 0
        tokens_out = resp.usage_metadata.candidates_token_count if resp.usage_metadata else 0

        logger.debug(
            f"Gemini call: model={self._model} tokens_in={tokens_in} tokens_out={tokens_out}"
        )
        return parsed, tokens_in, tokens_out
