from mtagentbench.swebench import OfficialSWEHarnessGrader, SWEHarnessResult


class FakeHarness(OfficialSWEHarnessGrader):
    calls = 0

    def _run(self, instance_id: str, patch: str) -> SWEHarnessResult:
        self.calls += 1
        return SWEHarnessResult(instance_id, True, patch_apply_ok=True)


def test_empty_patch_is_incorrect_not_infrastructure(tmp_path) -> None:
    result = OfficialSWEHarnessGrader(tmp_path).verify_patch("owner__repo-1", "")
    assert not result.resolved
    assert result.harness_error is None
    assert result.patch_extracted is False


def test_successful_official_result_is_cached_by_patch(tmp_path) -> None:
    grader = FakeHarness(tmp_path)
    patch = "diff --git a/a.py b/a.py\n@@ -1 +1 @@\n-a\n+b\n"
    first = grader.verify_patch("owner__repo-1", patch)
    second = grader.verify_patch("owner__repo-1", patch)
    assert first.resolved and second.resolved
    assert grader.calls == 1
    assert second.from_cache
