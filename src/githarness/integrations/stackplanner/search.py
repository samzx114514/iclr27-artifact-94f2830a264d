"""Read-only local corpus adapter for StackPlanner Search."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any
from urllib import error, request


_TOKEN = re.compile(r"[\w]+", re.UNICODE)


def _tokens(text: str) -> list[str]:
    return [token.casefold() for token in _TOKEN.findall(text)]


@dataclass(frozen=True)
class CorpusDocument:
    docid: str
    text: str
    title: str = ""


class JsonlSearchBackend:
    """Small deterministic BM25 retriever for local fixtures.

    Each JSONL row must expose ``docid`` and ``text``.  The class intentionally
    offers only ``search(query)``: the agent cannot inspect paths, index state,
    benchmark metadata, or hidden answers through this interface.
    """

    def __init__(self, corpus: Path, *, top_k: int = 8) -> None:
        self.top_k = top_k
        rows: list[CorpusDocument] = []
        for line_number, line in enumerate(Path(corpus).read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict) or not payload.get("docid") or not payload.get("text"):
                raise ValueError(f"invalid public corpus row {line_number}")
            rows.append(
                CorpusDocument(
                    docid=str(payload["docid"]),
                    text=str(payload["text"]),
                    title=str(payload.get("title", "")),
                )
            )
        if not rows:
            raise ValueError("public search corpus is empty")
        self.documents = tuple(rows)
        self._terms = [Counter(_tokens(f"{row.title} {row.text}")) for row in rows]
        self._lengths = [sum(counts.values()) for counts in self._terms]
        self._average_length = sum(self._lengths) / len(self._lengths)
        document_frequency: Counter[str] = Counter()
        for counts in self._terms:
            document_frequency.update(counts.keys())
        self._document_frequency = document_frequency

    def search(self, query: str) -> list[dict[str, Any]]:
        terms = _tokens(query)
        if not terms:
            return []
        scored = []
        for document, frequencies, length in zip(self.documents, self._terms, self._lengths):
            score = sum(self._bm25(term, frequencies, length) for term in terms)
            if score > 0:
                scored.append((score, document))
        scored.sort(key=lambda item: (-item[0], item[1].docid))
        return [
            {
                "docid": document.docid,
                "title": document.title,
                "snippet": document.text[:1200],
                "score": round(score, 6),
            }
            for score, document in scored[: self.top_k]
        ]

    def _bm25(self, term: str, frequencies: Counter[str], length: int) -> float:
        frequency = frequencies.get(term, 0)
        if frequency == 0:
            return 0.0
        total = len(self.documents)
        frequency_in_documents = self._document_frequency[term]
        inverse_document_frequency = math.log(1 + (total - frequency_in_documents + 0.5) / (frequency_in_documents + 0.5))
        denominator = frequency + 1.2 * (1 - 0.75 + 0.75 * length / self._average_length)
        return inverse_document_frequency * frequency * 2.2 / denominator


class RemoteBrowseCompSearchBackend:
    """Read-only client for the fixed BrowseComp-Plus retrieval service.

    The URL and credential are runtime inputs.  They are deliberately not
    persisted in traces or included in this object's error messages.
    """

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str = "",
        top_k: int = 5,
        snippet_max_tokens: int = 512,
        timeout: float = 120.0,
    ) -> None:
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("BrowseComp base URL must be non-empty")
        if not 1 <= top_k <= 20:
            raise ValueError("top_k must be between 1 and 20")
        if not 1 <= snippet_max_tokens <= 2048:
            raise ValueError("snippet_max_tokens must be between 1 and 2048")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        normalized = base_url.rstrip("/")
        self._endpoint = normalized if normalized.endswith("/search") else f"{normalized}/search"
        self._api_key = api_key
        self.top_k = top_k
        self.snippet_max_tokens = snippet_max_tokens
        self.timeout = timeout

    def search(self, query: str) -> list[dict[str, Any]]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("search query must be non-empty")
        body = json.dumps(
            {
                "query": query,
                "top_k": self.top_k,
                "snippet_max_tokens": self.snippet_max_tokens,
            }
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        call = request.Request(self._endpoint, data=body, headers=headers, method="POST")
        try:
            with request.urlopen(call, timeout=self.timeout) as response:
                payload: Any = json.load(response)
        except error.HTTPError as exc:
            raise RuntimeError(f"BrowseComp service returned HTTP {exc.code}") from exc
        except (error.URLError, TimeoutError) as exc:
            raise RuntimeError("BrowseComp service is unreachable") from exc

        rows = payload.get("hits") if isinstance(payload, dict) else None
        if not isinstance(rows, list) or len(rows) > self.top_k:
            raise ValueError("BrowseComp service returned an invalid hits array")
        hits: list[dict[str, Any]] = []
        for expected_rank, row in enumerate(rows, 1):
            if not isinstance(row, dict) or row.get("rank") != expected_rank:
                raise ValueError("BrowseComp service returned invalid ranks")
            docid, snippet, score = row.get("docid"), row.get("snippet"), row.get("score")
            if not isinstance(docid, str) or not docid or not isinstance(snippet, str):
                raise ValueError("BrowseComp hit lacks docid or snippet")
            if not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(score):
                raise ValueError("BrowseComp hit has an invalid score")
            hits.append({"docid": docid, "score": float(score), "snippet": snippet})
        return hits
