"""Official SWE-bench harness adapter derived from the authoritative remote tree."""

from __future__ import annotations

import contextlib
from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Iterator, Mapping


DEFAULT_DATASET = "princeton-nlp/SWE-bench_Verified"


@dataclass(frozen=True)
class SWEHarnessResult:
    instance_id: str
    resolved: bool
    patch_extracted: bool = True
    patch_apply_ok: bool | None = None
    ftp_pass: tuple[str, ...] = ()
    ftp_fail: tuple[str, ...] = ()
    ptp_pass: tuple[str, ...] = ()
    ptp_fail: tuple[str, ...] = ()
    harness_error: str | None = None
    duration_seconds: float = 0.0
    from_cache: bool = False
    raw_report: Mapping[str, Any] = field(default_factory=dict)

    def grader_mapping(self) -> dict[str, Any]:
        value = asdict(self)
        value["infrastructure_failure"] = self.harness_error is not None
        return value


class OfficialSWEHarnessGrader:
    """Grade one saved patch with the official SWE-bench evaluation library.

    Official datasets, Docker images, logs, and cache files remain external to
    the repository.  Import and harness failures are returned as infrastructure
    failures so callers remove them from correctness denominators.
    """

    _process_lock = threading.Lock()

    def __init__(
        self,
        workspace: Path,
        *,
        dataset_name: str = DEFAULT_DATASET,
        split: str = "test",
        timeout_seconds: int = 1800,
        cache_level: str = "env",
        namespace: str | None = "swebench",
        use_cache: bool = True,
    ) -> None:
        self.workspace = Path(workspace)
        self.cache_dir = self.workspace / "cache"
        self.predictions_dir = self.workspace / "predictions"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_name = dataset_name
        self.split = split
        self.timeout_seconds = timeout_seconds
        self.cache_level = cache_level
        self.namespace = namespace
        self.use_cache = use_cache

    def __call__(self, instance_id: str, patch: str) -> Mapping[str, Any]:
        return self.verify_patch(instance_id, patch).grader_mapping()

    def verify_patch(self, instance_id: str, patch: str) -> SWEHarnessResult:
        if not patch.strip():
            return SWEHarnessResult(
                instance_id,
                False,
                patch_extracted=False,
                patch_apply_ok=False,
            )
        cached = self._read_cache(instance_id, patch) if self.use_cache else None
        if cached is not None:
            return cached
        started = time.monotonic()
        result = self._run(instance_id, patch)
        result = SWEHarnessResult(
            **{
                **asdict(result),
                "duration_seconds": time.monotonic() - started,
            }
        )
        if self.use_cache and result.harness_error is None:
            self._write_cache(instance_id, patch, result)
        return result

    @staticmethod
    def _cache_key(instance_id: str, patch: str) -> str:
        patch_digest = hashlib.sha256(patch.encode("utf-8")).hexdigest()[:20]
        return f"{instance_id.replace('/', '__')}__{patch_digest}.json"

    def _cache_path(self, instance_id: str, patch: str) -> Path:
        return self.cache_dir / self._cache_key(instance_id, patch)

    def _read_cache(self, instance_id: str, patch: str) -> SWEHarnessResult | None:
        path = self._cache_path(instance_id, patch)
        if not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            value["from_cache"] = True
            for key in ("ftp_pass", "ftp_fail", "ptp_pass", "ptp_fail"):
                value[key] = tuple(value.get(key) or ())
            return SWEHarnessResult(**value)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def _write_cache(self, instance_id: str, patch: str, result: SWEHarnessResult) -> None:
        path = self._cache_path(instance_id, patch)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)

    def _run(self, instance_id: str, patch: str) -> SWEHarnessResult:
        try:
            from swebench.harness.constants import LOG_REPORT, RUN_EVALUATION_LOG_DIR
            from swebench.harness.run_evaluation import main as swebench_main
        except Exception as error:
            return self._infrastructure(instance_id, f"swebench_import_failed: {error}")

        patch_digest = hashlib.sha256(patch.encode("utf-8")).hexdigest()[:12]
        run_id = f"githarness_{patch_digest}"
        model_name = "githarness"
        prediction_directory = self.predictions_dir / run_id
        prediction_directory.mkdir(parents=True, exist_ok=True)
        predictions = prediction_directory / "predictions.json"
        predictions.write_text(
            json.dumps(
                [{"instance_id": instance_id, "model_name_or_path": model_name, "model_patch": patch}],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        report_relative = RUN_EVALUATION_LOG_DIR / run_id / model_name / instance_id / LOG_REPORT
        with self._process_lock, _pushd(self.workspace):
            try:
                swebench_main(
                    dataset_name=self.dataset_name,
                    split=self.split,
                    instance_ids=[instance_id],
                    predictions_path=str(predictions),
                    max_workers=1,
                    force_rebuild=False,
                    cache_level=self.cache_level,
                    clean=False,
                    open_file_limit=4096,
                    run_id=run_id,
                    timeout=self.timeout_seconds,
                    namespace=self.namespace,
                    rewrite_reports=False,
                    modal=False,
                )
            except BaseException as error:
                if isinstance(error, KeyboardInterrupt):
                    raise
                return self._infrastructure(instance_id, f"swebench_harness_failed: {type(error).__name__}: {error}")
        report_path = self.workspace / report_relative
        if not report_path.is_file():
            return self._infrastructure(instance_id, "swebench_report_not_written")
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            instance = report[instance_id]
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
            return self._infrastructure(instance_id, f"swebench_report_invalid: {error}")
        statuses = instance.get("tests_status") or {}
        fail_to_pass = statuses.get("FAIL_TO_PASS") or {}
        pass_to_pass = statuses.get("PASS_TO_PASS") or {}
        return SWEHarnessResult(
            instance_id=instance_id,
            resolved=bool(instance.get("resolved", False)),
            patch_apply_ok=bool(instance.get("patch_successfully_applied", False)),
            ftp_pass=tuple(fail_to_pass.get("success") or ()),
            ftp_fail=tuple(fail_to_pass.get("failure") or ()),
            ptp_pass=tuple(pass_to_pass.get("success") or ()),
            ptp_fail=tuple(pass_to_pass.get("failure") or ()),
            raw_report=dict(instance),
        )

    @staticmethod
    def _infrastructure(instance_id: str, message: str) -> SWEHarnessResult:
        return SWEHarnessResult(instance_id, False, harness_error=message)


@contextlib.contextmanager
def _pushd(target: Path) -> Iterator[None]:
    previous = Path.cwd()
    target.mkdir(parents=True, exist_ok=True)
    os.chdir(target)
    try:
        yield
    finally:
        os.chdir(previous)
