import hashlib
from typing import Any, Mapping

from .base import UpdateDomainContract

CONTRACT = UpdateDomainContract("research", "report.md", ("search(query)",), ("search",))
MAX_QUERIES = 8
MAX_HITS = 20


def register_hits(
    query: str,
    value: Any,
    ledger: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) > MAX_HITS:
        raise ValueError("research search capability returned invalid hits")
    by_source = {item["source_id"]: item for item in ledger}
    current = []
    for expected_rank, raw in enumerate(value, 1):
        if not isinstance(raw, Mapping):
            raise ValueError("research search hit must be an object")
        url = raw.get("url", raw.get("link", ""))
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            continue
        title = raw.get("title", raw.get("name", url))
        snippet = raw.get("snippet", raw.get("summary", raw.get("content", "")))
        date = raw.get("date", raw.get("datePublished", ""))
        if not isinstance(title, str) or not isinstance(snippet, str):
            raise ValueError("research search hit text is invalid")
        if not isinstance(date, str):
            date = ""
        source_id = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
        row = by_source.get(source_id)
        appearance = {"query": query, "rank": expected_rank}
        if row is None:
            row = {
                "label": len(ledger) + 1,
                "source_id": source_id,
                "title": title.strip() or url,
                "url": url,
                "snippet": snippet[:4000],
                "date": date,
                "queries": [appearance],
            }
            ledger.append(row)
            by_source[source_id] = row
        elif appearance not in row["queries"]:
            row["queries"].append(appearance)
        current.append(dict(row))
    return current
