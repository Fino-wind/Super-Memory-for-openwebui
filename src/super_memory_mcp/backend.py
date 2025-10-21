"""Persistence layer used by the Super Memory MCP server."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import aiohttp

from .models import MemoryRecord


class MemoryBackend:
    """Abstract persistence contract."""

    async def add_memory(self, record: MemoryRecord) -> str:
        raise NotImplementedError

    async def list_memories(self, user_id: str) -> List[MemoryRecord]:
        raise NotImplementedError

    async def delete_memories(self, memory_ids: Iterable[str]) -> None:
        raise NotImplementedError


class LocalJSONBackend(MemoryBackend):
    """Persistence backend that stores memories in a JSON file.

    This backend is primarily meant for development and automated testing. The
    file format is intentionally straightforward: ``{"memories": [...]}`` where
    each item mirrors :class:`MemoryRecord.as_dict` and includes a generated
    ``id`` field.
    """

    def __init__(self, path: Path):
        self._path = path
        self._lock = asyncio.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._path.write_text(
                json.dumps({"memories": []}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    async def _load(self) -> Dict[str, List[Dict[str, str]]]:
        async with self._lock:
            text = await asyncio.to_thread(self._path.read_text, encoding="utf-8")
        return json.loads(text)

    async def _dump(self, data: Dict[str, List[Dict[str, str]]]) -> None:
        payload = json.dumps(data, ensure_ascii=False, indent=2)
        async with self._lock:
            await asyncio.to_thread(self._path.write_text, payload, encoding="utf-8")

    async def add_memory(self, record: MemoryRecord) -> str:
        data = await self._load()
        memory_id = f"local-{int(datetime.now(tz=timezone.utc).timestamp() * 1000)}"
        payload = {"id": memory_id, **record.as_dict()}
        data.setdefault("memories", []).append(payload)
        await self._dump(data)
        return memory_id

    async def list_memories(self, user_id: str) -> List[MemoryRecord]:
        data = await self._load()
        results: List[MemoryRecord] = []
        for item in data.get("memories", []):
            if item.get("user_id") != user_id:
                continue
            created_at = datetime.fromisoformat(item["created_at"])
            results.append(
                MemoryRecord(
                    user_id=user_id,
                    fact=item["fact"],
                    created_at=created_at,
                    metadata=item.get("metadata", {}),
                )
            )
        return results

    async def delete_memories(self, memory_ids: Iterable[str]) -> None:
        ids = set(memory_ids)
        if not ids:
            return
        data = await self._load()
        data["memories"] = [item for item in data.get("memories", []) if item.get("id") not in ids]
        await self._dump(data)


class OpenWebUIBackend(MemoryBackend):
    """Persistence backend talking to the OpenWebUI REST API."""

    def __init__(self, base_url: str, token: str):
        self._base_url = base_url.rstrip("/")
        self._token = token

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self._token}", "Content-Type": "application/json"}

    async def add_memory(self, record: MemoryRecord) -> str:
        payload = record.as_dict()
        async with aiohttp.ClientSession(headers=self._headers()) as session:
            async with session.post(f"{self._base_url}/api/memory", json=payload, timeout=30) as response:
                response.raise_for_status()
                data = await response.json()
        return data.get("id", "")

    async def list_memories(self, user_id: str) -> List[MemoryRecord]:
        params = {"user_id": user_id}
        async with aiohttp.ClientSession(headers=self._headers()) as session:
            async with session.get(f"{self._base_url}/api/memory", params=params, timeout=30) as response:
                response.raise_for_status()
                items = await response.json()
        results: List[MemoryRecord] = []
        for item in items:
            created_at = datetime.fromisoformat(item["created_at"])
            results.append(
                MemoryRecord(
                    user_id=user_id,
                    fact=item["fact"],
                    created_at=created_at,
                    metadata=item.get("metadata", {}),
                )
            )
        return results

    async def delete_memories(self, memory_ids: Iterable[str]) -> None:
        ids = list(memory_ids)
        if not ids:
            return
        payload = {"ids": ids}
        async with aiohttp.ClientSession(headers=self._headers()) as session:
            async with session.delete(f"{self._base_url}/api/memory", json=payload, timeout=30) as response:
                response.raise_for_status()


def build_backend(storage_path: Path, base_url: Optional[str], token: Optional[str]) -> MemoryBackend:
    if base_url and token:
        return OpenWebUIBackend(base_url=base_url, token=token)
    return LocalJSONBackend(path=storage_path)
