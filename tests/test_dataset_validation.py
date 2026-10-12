import json

from mtagentbench.construction import build_dataset
from mtagentbench.contracts import to_plain
from mtagentbench.validation import validate_dataset, validate_search_corpus


def _write_release(tmp_path):
    source = [{"task_id": "m1", "turns": [{"query": "Compute 3 + 4"}], "label": "7"}]
    public, private, _ = build_dataset(source, domain="math")
    public_path = tmp_path / "public.jsonl"
    private_path = tmp_path / "private.jsonl"
    public_path.write_text(json.dumps(to_plain(public[0])) + "\n", encoding="utf-8")
    private_path.write_text(json.dumps(to_plain(private[0])) + "\n", encoding="utf-8")
    return public_path, private_path


def test_validate_paired_release(tmp_path) -> None:
    public, private = _write_release(tmp_path)
    report = validate_dataset(public, private)
    assert report["valid"]
    assert report["public_records"] == report["private_records"] == 1
    assert report["domains"] == {"math": 1}


def test_validate_detects_public_tampering(tmp_path) -> None:
    public, private = _write_release(tmp_path)
    value = json.loads(public.read_text(encoding="utf-8"))
    value["runtime_assets"]["gold_sql"] = "SELECT hidden"
    public.write_text(json.dumps(value) + "\n", encoding="utf-8")
    report = validate_dataset(public, private)
    assert not report["valid"]
    assert any("private fields" in error for error in report["errors"])


def test_search_corpus_validation_rejects_duplicate_docids(tmp_path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        '{"docid":"same","text":"first"}\n'
        '{"docid":"same","text":"second"}\n',
        encoding="utf-8",
    )
    count, errors = validate_search_corpus(corpus)
    assert count == 2
    assert errors == ["duplicate search docid: same"]


def test_search_corpus_validation_reports_non_object_rows(tmp_path) -> None:
    corpus = tmp_path / "corpus.json"
    corpus.write_text('["not-an-object"]', encoding="utf-8")
    count, errors = validate_search_corpus(corpus)
    assert count == 0
    assert len(errors) == 1 and errors[0].startswith("search corpus is invalid:")
