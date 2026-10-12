"""MTAgentBench data preparation CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .batch_evaluation import evaluate_files, load_callable, read_json_or_jsonl, write_evaluation
from .construction import build_dataset
from .contracts import to_plain
from .e2e_smoke import run_e2e_smoke
from .evaluator_smoke import run_smoke as run_evaluator_smoke
from .release import build_registry
from .release_validation import validate_release
from .search_assets import prepare_assets
from .sql_assets import enrich_public_schemas
from .dataset_builders.common import write_json
from .validation import validate_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mtagentbench", description="Prepare and evaluate MTAgentBench artifacts")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="split official source rows into public and private artifacts")
    prepare.add_argument("--domain", choices=("math", "sql", "search", "research", "code"), required=True)
    prepare.add_argument("--source", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument(
        "--db-root",
        type=Path,
        help="official BIRD dev_databases root used to fill absent public SQL schemas",
    )
    evaluate = sub.add_parser("evaluate", help="evaluate predictions using private records")
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--private-records", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--db-root", type=Path)
    evaluate.add_argument("--research-judge", help="module:function plugin")
    evaluate.add_argument("--code-grader", help="module:function plugin")
    evaluate.add_argument("--swebench-workspace", type=Path)
    evaluate.add_argument("--swebench-dataset", default="princeton-nlp/SWE-bench_Verified")
    validate = sub.add_parser("validate", help="validate paired public/private artifacts and manifest")
    validate.add_argument("--public", type=Path, required=True)
    validate.add_argument("--private-records", type=Path, required=True)
    validate.add_argument("--db-root", type=Path)
    validate.add_argument("--search-corpus", type=Path)
    validate.add_argument("--check-swe-images", action="store_true")
    validate.add_argument("--report", type=Path)
    release_registry = sub.add_parser(
        "release-registry", help="regenerate the canonical public dataset registry",
    )
    release_registry.add_argument("--root", type=Path, default=Path("."))
    release_registry.add_argument("--output", type=Path, default=Path("datasets/registry.json"))
    release_validate = sub.add_parser(
        "release-validate", help="validate all canonical public dataset splits",
    )
    release_validate.add_argument("--root", type=Path, default=Path("."))
    release_validate.add_argument("--registry", type=Path, default=Path("datasets/registry.json"))
    release_validate.add_argument("--require-private", action="store_true")
    search_assets = sub.add_parser(
        "prepare-search-assets", help="download or inventory pinned Search assets",
    )
    search_assets.add_argument("--manifest", type=Path, default=Path("datasets/search/retrieval_assets.json"))
    search_assets.add_argument("--output", type=Path, default=Path(".local/search-assets"))
    search_assets.add_argument("--download", action="store_true")
    search_assets.add_argument("--hash-files", action="store_true")
    smoke = sub.add_parser("smoke", help="run the five-domain offline end-to-end smoke")
    smoke.add_argument("--output", type=Path)
    evaluator_smoke = sub.add_parser(
        "smoke-evaluators", help="run the five offline evaluator adapter smokes",
    )
    evaluator_smoke.add_argument("--workspace", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "release-registry":
        root = args.root.resolve()
        output = args.output if args.output.is_absolute() else root / args.output
        registry = build_registry(root)
        write_json(output, registry)
        print(json.dumps({"output": str(output), "splits": len(registry["splits"])}))
        return 0
    if args.command == "release-validate":
        root = args.root.resolve()
        registry_path = args.registry if args.registry.is_absolute() else root / args.registry
        result = validate_release(
            root, registry_path=registry_path, require_private=args.require_private,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if result["valid"] else 1
    if args.command == "prepare-search-assets":
        receipt = prepare_assets(
            args.manifest.resolve(), args.output.resolve(),
            download=args.download, hash_files=args.hash_files,
        )
        print(receipt)
        return 0
    if args.command == "smoke":
        if args.output is None:
            import tempfile

            with tempfile.TemporaryDirectory(prefix="githarness-dataset-e2e-") as value:
                report = run_e2e_smoke(Path(value))
        else:
            report = run_e2e_smoke(args.output.resolve())
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if report["valid"] else 1
    if args.command == "smoke-evaluators":
        if args.workspace is None:
            import tempfile

            with tempfile.TemporaryDirectory(prefix="githarness-evaluator-smoke-") as value:
                results = run_evaluator_smoke(Path(value))
        else:
            results = run_evaluator_smoke(args.workspace)
        report = {
            "schema_version": 1,
            "valid": all(result["status"] == "correct" for result in results),
            "domains": [result["domain"] for result in results],
            "results": results,
            "external_services_called": False,
        }
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if report["valid"] else 1
    if args.command == "validate":
        report = validate_dataset(
            args.public,
            args.private_records,
            db_root=args.db_root,
            search_corpus=args.search_corpus,
            check_swe_images=args.check_swe_images,
        )
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["valid"] else 1
    if args.command == "evaluate":
        code_grader = load_callable(args.code_grader)
        if code_grader is None and args.swebench_workspace is not None:
            from .swebench import OfficialSWEHarnessGrader

            code_grader = OfficialSWEHarnessGrader(
                args.swebench_workspace,
                dataset_name=args.swebench_dataset,
            )
        results, summary = evaluate_files(
            args.predictions,
            args.private_records,
            db_root=args.db_root,
            research_judge=load_callable(args.research_judge),
            code_grader=code_grader,
        )
        write_evaluation(args.output, results, summary)
        print(json.dumps(summary, ensure_ascii=False))
        return 0
    source_rows = read_json_or_jsonl(args.source)
    rows = enrich_public_schemas(source_rows, args.db_root) if args.domain == "sql" else source_rows
    public, private, exclusions = build_dataset(rows, domain=args.domain)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, values in (("public_cases.jsonl", public), ("private_eval.jsonl", private)):
        (args.output / name).write_text("".join(json.dumps(to_plain(value), ensure_ascii=False) + "\n" for value in values), encoding="utf-8")
    print(json.dumps({
        "selected": len(public),
        "excluded": len(exclusions),
        "output": str(args.output),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
