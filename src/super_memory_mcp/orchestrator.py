"""High level orchestration for conversation processing."""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import List, Mapping, Optional, Sequence

import numpy as np
import pytz

from .backend import MemoryBackend
from .config import PluginConfig
from .embeddings import embed_texts
from .llm import ChatCompletionClient
from .models import ConversationTurn, MemoryRecord, ProcessingStats, tail_messages


class MemoryOrchestrator:
    """Coordinates LLM calls, memory persistence and statistics collection."""

    def __init__(self, config: PluginConfig, backend: MemoryBackend, llm: ChatCompletionClient) -> None:
        self._config = config
        self._backend = backend
        self._llm = llm

    async def process_conversation(
        self,
        user_id: str,
        conversation: Sequence[ConversationTurn],
    ) -> Mapping[str, object]:
        if not self._config.enabled:
            return {"status": "disabled"}
        start_time = time.perf_counter()
        recent_messages = tail_messages(conversation, self._config.messages_to_consider)
        llm_messages = [
            {"role": turn.role, "content": turn.content}
            for turn in recent_messages
        ]

        created_records: List[MemoryRecord] = []
        llm_tokens = 0
        ttfb = None

        if self._config.memory_mode == "fact":
            created_records, usage = await self._extract_facts(user_id, llm_messages)
            if usage:
                llm_tokens = usage.get("total_tokens") or (
                    usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
                )
        else:
            summary, usage = await self._create_summary(user_id, llm_messages)
            if summary:
                created_records = [summary]
            if usage:
                llm_tokens = usage.get("total_tokens") or (
                    usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
                )

        for record in created_records:
            await self._backend.add_memory(record)

        total_time = time.perf_counter() - start_time
        stats = ProcessingStats(
            total_duration=total_time,
            time_to_first_token=ttfb,
            total_tokens=llm_tokens or None,
            tokens_per_second=(llm_tokens / total_time) if llm_tokens and total_time else None,
        )

        return {
            "status": "ok",
            "created": [record.as_dict() for record in created_records],
            "stats": stats.as_dict() if self._config.show_stats else None,
        }

    async def _extract_facts(
        self,
        user_id: str,
        llm_messages: Sequence[Mapping[str, str]],
    ) -> tuple[List[MemoryRecord], Optional[dict[str, int]]]:
        candidates, usage = await self._llm.extract_facts(llm_messages)
        if not candidates:
            return [], usage

        existing = await self._backend.list_memories(user_id)
        to_persist: List[MemoryRecord] = []
        tz = pytz.timezone(self._config.timezone)
        now = datetime.now(tz=tz)

        for fact in candidates:
            if not await self._is_new_fact(fact, existing):
                continue
            record = MemoryRecord(
                user_id=user_id,
                fact=fact,
                created_at=now,
                metadata={"mode": "fact"},
            )
            to_persist.append(record)
        return to_persist, usage

    async def _is_new_fact(self, fact: str, existing: Sequence[MemoryRecord]) -> bool:
        for record in existing:
            if record.fact == fact:
                return False
        if not self._config.enable_semantic_search:
            return True
        return await self._semantic_check(fact, existing)

    async def _semantic_check(self, fact: str, existing: Sequence[MemoryRecord]) -> bool:
        if not existing:
            return True
        embeddings = await embed_texts([fact, *[item.fact for item in existing]])
        fact_vector = np.array(embeddings[0])
        existing_vectors = np.array(embeddings[1:])
        similarities = existing_vectors @ fact_vector / (
            np.linalg.norm(existing_vectors, axis=1) * np.linalg.norm(fact_vector)
        )
        return bool(np.all(similarities < self._config.similarity_threshold))

    async def _create_summary(
        self,
        user_id: str,
        llm_messages: Sequence[Mapping[str, str]],
    ) -> tuple[MemoryRecord | None, Optional[dict[str, int]]]:
        conversation_text = "\n".join(turn["content"] for turn in llm_messages)
        summary_text, usage = await self._llm.create_summary(conversation_text)
        if not summary_text:
            return None, usage
        tz = pytz.timezone(self._config.timezone)
        return MemoryRecord(
            user_id=user_id,
            fact=summary_text,
            created_at=datetime.now(tz=tz),
            metadata={"mode": "summary"},
        ), usage

    async def summarise_clusters(self, user_id: str) -> Mapping[str, object]:
        memories = await self._backend.list_memories(user_id)
        if not memories:
            return {"status": "ok", "summaries": []}

        clusters = await self._cluster_memories(memories)
        summaries = []
        for cluster in clusters:
            summary_text, _usage = await self._llm.summarize_cluster([item.fact for item in cluster])
            if not summary_text:
                continue
            tz = pytz.timezone(self._config.timezone)
            summaries.append(
                MemoryRecord(
                    user_id=user_id,
                    fact=summary_text,
                    created_at=datetime.now(tz=tz),
                    metadata={
                        "mode": "summary_cluster",
                        "source_facts": [item.fact for item in cluster],
                    },
                )
            )
        return {"status": "ok", "summaries": [item.as_dict() for item in summaries]}

    async def _cluster_memories(self, memories: Sequence[MemoryRecord]) -> List[List[MemoryRecord]]:
        if len(memories) < self._config.summarization_min_cluster_size:
            return []
        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=self._config.summarization_min_memory_age_days)
        candidates = [
            item
            for item in memories
            if item.created_at.astimezone(timezone.utc) < cutoff
        ]
        if len(candidates) < self._config.summarization_min_cluster_size:
            return []
        embeddings = await embed_texts([item.fact for item in candidates])
        matrix = np.array(embeddings)
        clusters: List[List[MemoryRecord]] = []
        used = set()
        for idx, vector in enumerate(matrix):
            if idx in used:
                continue
            cluster = [candidates[idx]]
            used.add(idx)
            similarities = matrix @ vector / (
                np.linalg.norm(matrix, axis=1) * np.linalg.norm(vector)
            )
            for jdx, score in enumerate(similarities):
                if jdx == idx or jdx in used:
                    continue
                if score >= self._config.summarization_cluster_threshold:
                    cluster.append(candidates[jdx])
                    used.add(jdx)
            if len(cluster) >= self._config.summarization_min_cluster_size:
                clusters.append(cluster)
        return clusters
