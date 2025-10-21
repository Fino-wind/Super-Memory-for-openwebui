"""LLM client helpers used by the Super Memory MCP server."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import aiohttp

from .prompts import (
    FACT_CONSOLIDATION_PROMPT,
    FACT_EXTRACTION_PROMPT,
    MEMORY_SUMMARIZATION_PROMPT,
    SUMMARY_CREATION_PROMPT,
    SUMMARY_UPDATE_PROMPT,
)


@dataclass(slots=True)
class LLMResponse:
    content: str
    raw: Mapping[str, object]
    usage: Optional[Dict[str, int]]


class ChatCompletionClient:
    """Minimal OpenAI-compatible chat completion helper."""

    def __init__(self, api_url: str, api_key: str, model: str) -> None:
        self._api_url = api_url
        self._api_key = api_key
        self._model = model

    async def _request(self, messages: List[Mapping[str, str]], **extra: object) -> LLMResponse:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        payload = {"model": self._model, "messages": messages, **extra}
        async with aiohttp.ClientSession() as session:
            async with session.post(self._api_url, json=payload, headers=headers, timeout=120) as response:
                response.raise_for_status()
                data = await response.json()
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError("LLM response did not contain choices")
        message = choices[0].get("message", {})
        content = message.get("content", "")
        usage = data.get("usage")
        return LLMResponse(content=content, raw=data, usage=usage)

    async def extract_facts(
        self, messages: Iterable[Mapping[str, str]]
    ) -> Tuple[List[str], Optional[Dict[str, int]]]:
        payload = list(messages)
        response = await self._request(
            [
                {"role": "system", "content": FACT_EXTRACTION_PROMPT},
                *payload,
            ],
            temperature=0.1,
        )
        try:
            facts = json.loads(response.content)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Failed to decode fact extraction response") from exc
        return facts, response.usage

    async def create_summary(
        self, conversation: str
    ) -> Tuple[str, Optional[Dict[str, int]]]:
        response = await self._request(
            [
                {"role": "system", "content": SUMMARY_CREATION_PROMPT},
                {"role": "user", "content": conversation},
            ],
            temperature=0.2,
        )
        return response.content.strip(), response.usage

    async def update_summary(
        self, existing: str, conversation: str
    ) -> Tuple[str, Optional[Dict[str, int]]]:
        prompt = SUMMARY_UPDATE_PROMPT.format(
            existing_summary=existing,
            new_conversation=conversation,
        )
        response = await self._request(
            [
                {"role": "system", "content": prompt},
            ],
            temperature=0.2,
        )
        return response.content.strip(), response.usage

    async def consolidate_facts(
        self, items: List[Mapping[str, object]]
    ) -> Tuple[List[str], Optional[Dict[str, int]]]:
        response = await self._request(
            [
                {"role": "system", "content": FACT_CONSOLIDATION_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(items, ensure_ascii=False),
                },
            ],
            temperature=0,
        )
        try:
            facts = json.loads(response.content)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Failed to decode consolidation response") from exc
        return facts, response.usage

    async def summarize_cluster(
        self, memories: List[str]
    ) -> Tuple[str, Optional[Dict[str, int]]]:
        response = await self._request(
            [
                {"role": "system", "content": MEMORY_SUMMARIZATION_PROMPT},
                {"role": "user", "content": "\n".join(memories)},
            ],
            temperature=0.2,
        )
        return response.content.strip(), response.usage
