"""Common data structures used throughout the Super Memory MCP server."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional


@dataclass(slots=True)
class MemoryRecord:
    """A single factual memory captured for a user."""

    user_id: str
    fact: str
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "fact": self.fact,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class MemorySummary:
    """Condensed representation of multiple memories."""

    user_id: str
    summary: str
    created_at: datetime
    source_ids: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "summary": self.summary,
            "created_at": self.created_at.isoformat(),
            "source_ids": list(self.source_ids),
        }


@dataclass(slots=True)
class ProcessingStats:
    """Metrics emitted after each conversation processing run."""

    total_duration: float
    time_to_first_token: Optional[float] = None
    total_tokens: Optional[int] = None
    tokens_per_second: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "total_duration": self.total_duration,
            "time_to_first_token": self.time_to_first_token,
            "total_tokens": self.total_tokens,
            "tokens_per_second": self.tokens_per_second,
        }


@dataclass(slots=True)
class ConversationTurn:
    role: str
    content: str


def tail_messages(messages: Iterable[ConversationTurn], limit: int) -> List[ConversationTurn]:
    stack = list(messages)
    return stack[-limit:]
