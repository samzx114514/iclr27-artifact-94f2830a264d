import math
from typing import Any, Mapping

from .base import UpdateDomainContract

CONTRACT = UpdateDomainContract("search", "answer.json", ("search(query)",), ("search",))
MAX_QUERIES = 4
MAX_HITS = 100


def normalize_hits(query: str, value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) > MAX_HITS:
        raise ValueError("search capability returned invalid hits")
    rows = []
    for expected_rank, raw in enumerate(value, 1):
        if not isinstance(raw, Mapping):
            raise ValueError("search hit must be an object")
        docid = raw.get("docid")
        rank = raw.get("rank", expected_rank)
        score = raw.get("score", 0.0)
        snippet = raw.get("snippet", "")
        if not isinstance(docid, str) or not docid:
            raise ValueError("search hit docid is required")
        if isinstance(rank, bool) or rank != expected_rank:
            raise ValueError("search hit ranks must be consecutive")
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
        ):
            raise ValueError("search hit score must be finite")
        if not isinstance(snippet, str):
            raise ValueError("search hit snippet must be a string")
        rows.append({
            "query": query,
            "docid": docid,
            "rank": rank,
            "score": float(score),
            "snippet": snippet,
        })
    return rows
