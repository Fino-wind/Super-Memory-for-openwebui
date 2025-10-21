"""Entry point for the Super Memory Model Context Protocol server."""
from __future__ import annotations

from typing import Dict, List

from mcp.server.fastmcp import FastMCP

from .backend import build_backend
from .config import PluginConfig, load_config
from .llm import ChatCompletionClient
from .models import ConversationTurn
from .orchestrator import MemoryOrchestrator

mcp = FastMCP("super-memory")


def _load_config() -> PluginConfig:
    # In the MCP world configuration is typically passed through environment
    # variables. The defaults defined in :mod:`config` align with the legacy
    # function based integration and provide backwards compatibility.
    return load_config()


def _build_orchestrator() -> MemoryOrchestrator:
    config = _load_config()
    backend = build_backend(
        storage_path=config.storage_path,
        base_url=config.openwebui_base_url,
        token=config.openwebui_token,
    )
    llm = ChatCompletionClient(config.api_url, config.api_key, config.model)
    return MemoryOrchestrator(config=config, backend=backend, llm=llm)


@mcp.tool()
async def process_conversation(user_id: str, messages: List[Dict[str, str]]) -> Dict[str, object]:
    """Process a conversation transcript and persist newly discovered memories."""

    orchestrator = _build_orchestrator()
    turns = [ConversationTurn(role=item["role"], content=item["content"]) for item in messages]
    result = await orchestrator.process_conversation(user_id=user_id, conversation=turns)
    return dict(result)


@mcp.tool()
async def summarise_clusters(user_id: str) -> Dict[str, object]:
    """Trigger clustering and summarisation of historical memories for the user."""

    orchestrator = _build_orchestrator()
    return dict(await orchestrator.summarise_clusters(user_id=user_id))


@mcp.tool()
async def status() -> Dict[str, object]:
    config = _load_config()
    return {
        "enabled": config.enabled,
        "mode": config.memory_mode,
        "semantic_search": config.enable_semantic_search,
    }


def run() -> None:
    orchestrator = _build_orchestrator()
    # Trigger background tasks when configured. For now we just ensure that the
    # backend is instantiated to catch configuration errors early.
    del orchestrator
    mcp.run()


if __name__ == "__main__":
    run()
