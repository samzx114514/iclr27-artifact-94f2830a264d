import json
from io import BytesIO
from urllib import error

import pytest

from githarness.integrations.stackplanner.research import (
    BochaInfrastructureError,
    RemoteBochaSearchBackend,
)
from githarness.integrations.stackplanner.search import (
    JsonlSearchBackend,
    RemoteBrowseCompSearchBackend,
)


def test_jsonl_search_backend_is_read_only_and_deterministic(tmp_path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    rows = [
        {"docid": "b", "title": "Other", "text": "A common unrelated passage."},
        {"docid": "a", "title": "Target", "text": "The rare cobalt comet was observed."},
    ]
    corpus.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    backend = JsonlSearchBackend(corpus, top_k=1)
    assert backend.search("rare cobalt")[0]["docid"] == "a"
    assert backend.search("rare cobalt") == backend.search("rare cobalt")


class _Response(BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def test_remote_browsecomp_backend_validates_and_does_not_leak_secret(monkeypatch) -> None:
    observed = {}

    def fake_urlopen(call, timeout):
        observed["url"] = call.full_url
        observed["authorization"] = call.headers["Authorization"]
        observed["body"] = json.loads(call.data)
        observed["timeout"] = timeout
        return _Response(json.dumps({"hits": [{"rank": 1, "docid": "d1", "score": 0.5, "snippet": "evidence"}]}).encode())

    monkeypatch.setattr("githarness.integrations.stackplanner.search.request.urlopen", fake_urlopen)
    backend = RemoteBrowseCompSearchBackend(
        "https://retriever.example/v1", api_key="private-value", top_k=5, snippet_max_tokens=512, timeout=9
    )
    assert backend.search("target clue") == [{"docid": "d1", "score": 0.5, "snippet": "evidence"}]
    assert observed == {
        "url": "https://retriever.example/v1/search",
        "authorization": "Bearer private-value",
        "body": {"query": "target clue", "top_k": 5, "snippet_max_tokens": 512},
        "timeout": 9,
    }
    assert "private-value" not in repr(backend.__dict__.get("_endpoint"))


def test_remote_browsecomp_backend_classifies_network_failure_without_url(monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise error.URLError("internal host detail")

    monkeypatch.setattr("githarness.integrations.stackplanner.search.request.urlopen", fail)
    backend = RemoteBrowseCompSearchBackend("https://retriever.example/v1")
    with pytest.raises(RuntimeError, match="service is unreachable") as caught:
        backend.search("query")
    assert "retriever.example" not in str(caught.value)


@pytest.mark.parametrize(
    "payload",
    [
        {"hits": [{"rank": 2, "docid": "d", "score": 1.0, "snippet": "x"}]},
        {"hits": [{"rank": 1, "docid": "d", "score": float("nan"), "snippet": "x"}]},
        {"hits": [{"rank": 1, "docid": "", "score": 1.0, "snippet": "x"}]},
    ],
)
def test_remote_browsecomp_backend_rejects_malformed_results(monkeypatch, payload) -> None:
    monkeypatch.setattr(
        "githarness.integrations.stackplanner.search.request.urlopen",
        lambda *args, **kwargs: _Response(json.dumps(payload).encode()),
    )
    with pytest.raises(ValueError):
        RemoteBrowseCompSearchBackend("https://retriever.example/v1").search("query")


def test_remote_bocha_backend_normalizes_public_hits_without_leaking_secret(monkeypatch) -> None:
    observed = {}

    def fake_urlopen(call, timeout):
        observed["authorization"] = call.headers["Authorization"]
        observed["body"] = json.loads(call.data)
        observed["timeout"] = timeout
        payload = {
            "code": 200,
            "data": {
                "webPages": {
                    "value": [{
                        "name": "Public title",
                        "url": "https://example.org/source",
                        "summary": "Public summary",
                        "datePublished": "2026-01-02",
                    }]
                }
            },
        }
        return _Response(json.dumps(payload).encode())

    monkeypatch.setattr(
        "githarness.integrations.stackplanner.research.request.urlopen",
        fake_urlopen,
    )
    backend = RemoteBochaSearchBackend(
        api_key="private-value", top_k=3, timeout=9, max_attempts=1
    )

    assert backend.search("public topic") == [{
        "title": "Public title",
        "url": "https://example.org/source",
        "snippet": "Public summary",
        "date": "2026-01-02",
    }]
    assert observed == {
        "authorization": "Bearer private-value",
        "body": {
            "query": "public topic",
            "freshness": "noLimit",
            "summary": True,
            "count": 3,
            "page": 1,
        },
        "timeout": 9,
    }
    assert "private-value" not in repr(backend)


def test_remote_bocha_backend_redacts_network_details(monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise error.URLError("secret internal route")

    monkeypatch.setattr(
        "githarness.integrations.stackplanner.research.request.urlopen", fail
    )
    backend = RemoteBochaSearchBackend(
        api_key="private-value", max_attempts=1, retry_backoff=0
    )
    with pytest.raises(BochaInfrastructureError, match="service is unreachable") as caught:
        backend.search("public query")
    assert "secret internal route" not in str(caught.value)
    assert "private-value" not in str(caught.value)


@pytest.mark.parametrize(
    "payload",
    [
        {"code": 403, "data": {}},
        {"code": 200, "data": {"webPages": {"value": "invalid"}}},
        {"code": 200, "data": {"webPages": {"value": ["invalid"]}}},
    ],
)
def test_remote_bocha_backend_rejects_malformed_results(monkeypatch, payload) -> None:
    monkeypatch.setattr(
        "githarness.integrations.stackplanner.research.request.urlopen",
        lambda *args, **kwargs: _Response(json.dumps(payload).encode()),
    )
    with pytest.raises(BochaInfrastructureError):
        RemoteBochaSearchBackend(
            api_key="private-value", max_attempts=1
        ).search("query")
