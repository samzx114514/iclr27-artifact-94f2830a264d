"""Narrow read-only BoCha provider for canonical Research execution."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import time
from typing import Any, Mapping
from urllib import error, parse, request


DEFAULT_BOCHA_ENDPOINT = "https://api.langsearch.com/v1/web-search"


class BochaInfrastructureError(RuntimeError):
    """The public BoCha provider failed without exposing connection details."""


@dataclass(frozen=True, slots=True)
class RemoteBochaSearchBackend:
    """Strict BoCha client exposing only ``search(query)`` and public hits."""

    api_key: str = field(repr=False)
    endpoint: str = DEFAULT_BOCHA_ENDPOINT
    top_k: int = 8
    timeout: float = 60.0
    max_attempts: int = 3
    retry_backoff: float = 3.0

    def __post_init__(self) -> None:
        endpoint = self.endpoint.strip() if isinstance(self.endpoint, str) else ""
        parsed = parse.urlparse(endpoint)
        if parsed.scheme != "https" or not parsed.hostname:
            raise ValueError("BoCha endpoint must be an HTTPS URL")
        if endpoint != DEFAULT_BOCHA_ENDPOINT:
            raise ValueError("BoCha endpoint must use the pinned public provider")
        if not isinstance(self.api_key, str) or not self.api_key:
            raise ValueError("BoCha credential must be non-empty")
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int) or not 1 <= self.top_k <= 20:
            raise ValueError("BoCha top_k must be between 1 and 20")
        if not isinstance(self.timeout, (int, float)) or self.timeout <= 0:
            raise ValueError("BoCha timeout must be positive")
        if isinstance(self.max_attempts, bool) or not isinstance(self.max_attempts, int) or self.max_attempts < 1:
            raise ValueError("BoCha max_attempts must be positive")
        if not isinstance(self.retry_backoff, (int, float)) or self.retry_backoff < 0:
            raise ValueError("BoCha retry_backoff must be non-negative")
        object.__setattr__(self, "endpoint", endpoint)

    def search(self, query: str) -> list[dict[str, Any]]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("BoCha query must be non-empty")
        body = json.dumps(
            {
                "query": query.strip(),
                "freshness": "noLimit",
                "summary": True,
                "count": self.top_k,
                "page": 1,
            },
            ensure_ascii=True,
        ).encode("utf-8")
        call = request.Request(
            self.endpoint,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        payload: Any = None
        last_error: BochaInfrastructureError | None = None
        for attempt in range(self.max_attempts):
            try:
                with request.urlopen(call, timeout=self.timeout) as response:
                    payload = json.load(response)
                last_error = None
                break
            except error.HTTPError as exc:
                last_error = BochaInfrastructureError(
                    f"BoCha service returned HTTP {exc.code}"
                )
                if exc.code != 429 and not 500 <= exc.code < 600:
                    raise last_error from exc
            except (error.URLError, TimeoutError, OSError) as exc:
                last_error = BochaInfrastructureError(
                    f"BoCha service is unreachable ({type(exc).__name__})"
                )
            except json.JSONDecodeError as exc:
                raise BochaInfrastructureError(
                    "BoCha service returned invalid JSON"
                ) from exc
            if attempt + 1 < self.max_attempts:
                time.sleep(self.retry_backoff * (attempt + 1))
        if last_error is not None:
            raise last_error
        return self._public_hits(payload)

    def _public_hits(self, payload: Any) -> list[dict[str, Any]]:
        if not isinstance(payload, Mapping):
            raise BochaInfrastructureError("BoCha service returned an invalid response")
        code = payload.get("code")
        if code is not None and code not in {200, "200"}:
            raise BochaInfrastructureError(f"BoCha service returned error code {code!r}")
        data = payload.get("data")
        if data is None:
            return []
        if not isinstance(data, Mapping):
            raise BochaInfrastructureError("BoCha service returned an invalid response")
        pages = data.get("webPages")
        if pages is None:
            return []
        if not isinstance(pages, Mapping):
            raise BochaInfrastructureError("BoCha service returned invalid web results")
        values = pages.get("value")
        if values is None:
            return []
        if not isinstance(values, list) or len(values) > self.top_k:
            raise BochaInfrastructureError("BoCha service returned invalid web results")

        hits: list[dict[str, Any]] = []
        for raw in values:
            if not isinstance(raw, Mapping):
                raise BochaInfrastructureError("BoCha service returned an invalid hit")
            url = raw.get("url", raw.get("link", ""))
            title = raw.get("name", raw.get("title", ""))
            snippet = raw.get("summary", raw.get("snippet", raw.get("content", "")))
            date = raw.get("datePublished", raw.get("dateLastCrawled", raw.get("date", "")))
            if not isinstance(url, str) or not url.startswith(("http://", "https://")):
                continue
            if not isinstance(title, str) or not isinstance(snippet, str):
                raise BochaInfrastructureError("BoCha service returned an invalid hit")
            hits.append(
                {
                    "title": title.strip() or url,
                    "url": url,
                    "snippet": snippet,
                    "date": date if isinstance(date, str) else "",
                }
            )
        return hits
