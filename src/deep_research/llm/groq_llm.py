"""Groq wrapper with structured output support."""

from __future__ import annotations

import asyncio
from typing import TypeVar

import groq
from groq import AsyncGroq
from loguru import logger
from pydantic import BaseModel

# ⚠️ อิมพอร์ต settings เพื่อดึง API Key ที่ Pydantic ตรวจสอบแล้ว
from deep_research.config import settings

T = TypeVar("T", bound=BaseModel)


class GroqLLM:
    """Groq wrapper. Enforces JSON schema compliance."""

    def __init__(self, model: str = "llama3-8b-8192") -> None:
        self._model = model
        # ⚠️ เรียกใช้กุญแจ API จาก settings และแกะ SecretStr ออกมา
        self._client = AsyncGroq(api_key=settings.GROQ_API_KEY.get_secret_value())

    async def structured_complete(
        self,
        *,
        system: str,
        user: str,
        response_model: type[T],
        max_tokens: int = 2048,
    ) -> tuple[T, int, int]:
        """Force JSON output matching response_model using tool calling."""

        schema = response_model.model_json_schema()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "submit_response",
                    "description": f"Submit the final {response_model.__name__}.",
                    "parameters": schema,
                },
            }
        ]

        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = await self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    tools=tools,
                    tool_choice={"type": "function", "function": {"name": "submit_response"}},
                    max_tokens=max_tokens,
                    temperature=0.2,
                )

                tool_call = resp.choices[0].message.tool_calls[0]
                parsed = response_model.model_validate_json(tool_call.function.arguments)

                tokens_in = resp.usage.prompt_tokens
                tokens_out = resp.usage.completion_tokens

                logger.debug(
                    f"Groq call: model={self._model} tokens_in={tokens_in} tokens_out={tokens_out}"
                )
                return parsed, tokens_in, tokens_out

            except (
                groq.BadRequestError,
                groq.RateLimitError,
                groq.APIConnectionError,
                groq.APITimeoutError,
            ) as exc:
                if attempt < max_retries - 1:
                    delay = 2**attempt  # 1s, 2s, 4s
                    logger.warning(
                        f"Groq transient error (attempt {attempt + 1}/{max_retries}), retrying in {delay}s: {exc}"
                    )
                    await asyncio.sleep(delay)
                else:
                    raise
