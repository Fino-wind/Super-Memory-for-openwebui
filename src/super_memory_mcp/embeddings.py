"""Embedding utilities used for semantic similarity checks."""
from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import Iterable, List


@lru_cache(maxsize=1)
def _load_model():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required for semantic search. Please install it via 'pip install sentence-transformers'."
        ) from exc
    return SentenceTransformer("all-MiniLM-L6-v2")


async def embed_texts(texts: Iterable[str]) -> List[List[float]]:
    loop = asyncio.get_event_loop()
    model = _load_model()
    return await loop.run_in_executor(None, lambda: model.encode(list(texts), convert_to_numpy=False))
