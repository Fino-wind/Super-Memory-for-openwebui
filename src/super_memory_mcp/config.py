"""Configuration handling for the Super Memory MCP server."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, validator


ENV_PREFIX = "SUPER_MEMORY_"


def _normalise_env_name(name: str) -> str:
    return ENV_PREFIX + name.upper()


class PluginConfig(BaseModel):
    """Runtime configuration loaded from environment variables or config files."""

    enabled: bool = Field(default=True, description="Whether the MCP tools are enabled.")
    memory_mode: Literal["fact", "summary"] = Field(
        default="fact", description="Memory extraction strategy that should be applied."
    )
    api_url: str = Field(
        default="https://api.openai.com/v1/chat/completions",
        description="OpenAI compatible chat completion endpoint.",
    )
    api_key: str = Field(default="", description="API key used for the LLM calls.")
    model: str = Field(default="gpt-4o-mini", description="Model used for memory processing.")
    show_stats: bool = Field(
        default=True, description="Expose performance metrics in MCP tool responses."
    )
    messages_to_consider: int = Field(
        default=6, description="Number of trailing messages taken into account for extraction."
    )
    timezone: str = Field(
        default="Asia/Shanghai",
        description="Timezone used to stamp memories.",
    )
    enable_semantic_search: bool = Field(
        default=False, description="Enable semantic similarity checks when deduplicating memories."
    )
    similarity_threshold: float = Field(
        default=0.8, description="Cosine similarity threshold used for de-duplication."
    )
    consolidation_threshold: float = Field(
        default=0.75,
        description="Threshold used to discover related facts for consolidation.",
    )
    enable_background_summarization: bool = Field(
        default=True,
        description="Whether periodic summarisation jobs should be created.",
    )
    summarization_interval_hours: float = Field(
        default=2.0, description="Interval between background summarisation runs."
    )
    summarization_cluster_threshold: float = Field(
        default=0.7, description="Similarity threshold when clustering facts for summaries."
    )
    summarization_min_cluster_size: int = Field(
        default=3, description="Minimum number of facts required to form a summary cluster."
    )
    summarization_min_memory_age_days: int = Field(
        default=7, description="Only memories older than this amount are summarised in background."
    )
    storage_path: Path = Field(
        default=Path("data/memories.json"),
        description="Location of the local JSON file used by the default backend.",
    )
    openwebui_base_url: Optional[str] = Field(
        default=None,
        description="Optional OpenWebUI instance base URL when delegating persistence.",
    )
    openwebui_token: Optional[str] = Field(
        default=None, description="Authentication token for the OpenWebUI REST API."
    )

    @validator("storage_path", pre=True)
    def _expand_storage_path(cls, value: Path | str) -> Path:
        if isinstance(value, Path):
            return value
        return Path(value).expanduser().resolve()

    @classmethod
    def from_env(cls) -> "PluginConfig":
        payload = {}
        for field in cls.model_fields:
            env_name = _normalise_env_name(field)
            if env_name in os.environ:
                payload[field] = os.environ[env_name]
        return cls.model_validate(payload)


DEFAULT_CONFIG = PluginConfig()


def load_config() -> PluginConfig:
    try:
        return PluginConfig.from_env()
    except Exception:
        return DEFAULT_CONFIG.model_copy()
