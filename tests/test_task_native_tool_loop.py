from __future__ import annotations

import json
import hashlib

import pytest

from githarness.domains import get_domain_spec
from githarness.domains.specs import validate_read_only_sql
from githarness.integrations.stackplanner.swe import CodeTestbedTaskNativeUpdateAgent
from githarness.task_native_update import (
    TaskNativePolicyResponse,
    TaskNativeToolCall,
    ToolLoopTaskNativeUpdateAgent,
    UpdateAgentRequest,
)
from githarness.task_native_update.domains.math import calculate_expression
from githarness.task_native_update.tools import task_native_tool_schemas
from githarness.version_interface import ExecutionResult
from githarness.version_interface import CandidateWorkState


class ScriptedPolicy:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, messages, tools):
        self.calls.append((list(messages), list(tools)))
        return self.responses.pop(0)


def call(name, arguments, index=0):
    return TaskNativePolicyResponse(
        tool_calls=(TaskNativeToolCall(f"call-{index}", name, arguments),),
    )


def request(tmp_path, domain, assets=None):
    spec = get_domain_spec(domain)
    return UpdateAgentRequest(
        task_id="task",
        domain=domain,
        candidate=CandidateWorkState("candidate", tmp_path, None),
        resolved_requirement="Complete public task",
        selected_base_checkpoint=None,
        update_hint="public hint",
        domain_spec=spec,
        public_runtime_assets={
            "capabilities": list(spec.public_capabilities),
            **(assets or {}),
        },
    )


class Search:
    def search(self, query):
        return [{"docid": "d1", "rank": 1, "score": 1.0, "snippet": f"evidence for {query}"}]


class ResearchSearch:
    def search(self, query):
        return [{
            "title": "Public source",
            "url": "https://example.invalid/source",
            "snippet": f"evidence for {query}",
            "date": "2026-01-01",
        }]


class FakeTestbed:
    def __init__(self):
        self.commands = []

    def execute(self, command):
        self.commands.append(command)
        return {"returncode": 0, "output": "done"}

    def git_diff(self):
        return "diff --git a/a.py b/a.py\n"


def test_update_agent_default_budget_is_paper_aligned() -> None:
    assert ToolLoopTaskNativeUpdateAgent(ScriptedPolicy([])).max_steps == 16
    with pytest.raises(ValueError, match="positive integer"):
        ToolLoopTaskNativeUpdateAgent(ScriptedPolicy([]), max_steps=0)
    with pytest.raises(ValueError, match="positive integer"):
        ToolLoopTaskNativeUpdateAgent(ScriptedPolicy([]), max_steps=True)


@pytest.mark.parametrize(
    ("domain", "assets", "responses", "artifact_path", "fragment"),
    [
        (
            "math",
            {},
            [
                call("calculate", {"expression": "2+5"}),
                call("write_artifact", {"content": "Reasoning. Final answer: \\boxed{7}"}, 1),
                call("finish", {}, 2),
            ],
            "answer.md",
            "boxed{7}",
        ),
        (
            "sql",
            {},
            [
                call("write_artifact", {"content": "SELECT title FROM books"}),
                call("finish", {}, 1),
            ],
            "answer.sql",
            "SELECT title",
        ),
        (
            "search",
            {"search": Search()},
            [
                call("search", {"query": "public clue"}),
                call("write_artifact", {"content": json.dumps({"exact_answer": "1997"})}, 1),
                call("finish", {}, 2),
            ],
            "answer.json",
            "1997",
        ),
        (
            "research",
            {"search": ResearchSearch()},
            [
                call("search", {"query": "public topic"}),
                call("write_artifact", {"content": "# Report\nEvidence【d1】"}, 1),
                call("finish", {}, 2),
            ],
            "report.md",
            "Report",
        ),
        (
            "code",
            {"testbed": FakeTestbed()},
            [
                call("testbed_execute", {"command": "python -m pytest -q"}),
                call("finish", {}, 1),
            ],
            "patch.diff",
            "diff --git",
        ),
    ],
)
def test_independent_tool_loop_supports_five_domains(
    tmp_path, domain, assets, responses, artifact_path, fragment
) -> None:
    policy = ScriptedPolicy(responses)
    result = ToolLoopTaskNativeUpdateAgent(policy, max_steps=6).execute(
        request(tmp_path, domain, assets)
    )

    assert result.finished
    assert fragment in result.artifact
    assert result.modified_work_state.files[artifact_path] == result.artifact
    tool_names = {tool["function"]["name"] for tool in policy.calls[0][1]}
    assert "finish" in tool_names
    assert {
        "glob_workspace",
        "grep_workspace",
        "read_workspace",
        "write_artifact",
        "patch_artifact",
        "delete_workspace",
    } <= tool_names
    assert ("search" in tool_names) is (domain in {"search", "research"})
    assert ("testbed_execute" in tool_names) is (domain == "code")
    assert "read_only_sql" not in tool_names


def test_workspace_inspect_and_explicit_delete_are_candidate_only(tmp_path) -> None:
    accepted = tmp_path / "accepted"
    candidate = tmp_path / "candidate"
    accepted.mkdir()
    candidate.mkdir()
    (accepted / "obsolete.md").write_text("needle in accepted", encoding="utf-8")
    (candidate / "answer.md").write_text("Current answer", encoding="utf-8")
    (candidate / "notes.md").write_text("needle in candidate", encoding="utf-8")
    (candidate / "obsolete.md").write_text("withdrawn file", encoding="utf-8")
    policy = ScriptedPolicy([
        call("glob_workspace", {"pattern": "**/*.md"}),
        call("grep_workspace", {"query": "needle", "pattern": "**/*.md"}, 1),
        call(
            "delete_workspace",
            {
                "path": "obsolete.md",
                "reason": "The complete requirement explicitly withdraws this appendix.",
            },
            2,
        ),
        call("finish", {}, 3),
    ])

    result = ToolLoopTaskNativeUpdateAgent(policy, max_steps=4).execute(
        request(candidate, "math")
    )

    assert result.finished
    assert result.trace[0]["observation"] == {
        "ok": True,
        "paths": ["answer.md", "notes.md", "obsolete.md"],
        "truncated": False,
    }
    assert result.trace[1]["observation"]["matches"] == [{
        "path": "notes.md",
        "line": 1,
        "text": "needle in candidate",
    }]
    assert result.trace[2]["observation"] == {
        "ok": True,
        "deleted": True,
        "path": "obsolete.md",
        "reason": "The complete requirement explicitly withdraws this appendix.",
    }
    assert "obsolete.md" not in result.modified_work_state.files
    assert (accepted / "obsolete.md").read_text(encoding="utf-8") == "needle in accepted"
    system_prompt = policy.calls[0][0][0]["content"]
    assert "explicitly withdraws an entire inherited file" in system_prompt


@pytest.mark.parametrize(
    "path",
    [
        "answer.md",
        "answer.sql",
        "answer.json",
        "report.md",
        "patch.diff",
        "evidence.jsonl",
        "references.json",
        "research_notes.jsonl",
        "test_receipt.json",
    ],
)
def test_delete_rejects_primary_and_runtime_owned_files(tmp_path, path: str) -> None:
    (tmp_path / path).write_text("protected", encoding="utf-8")
    policy = ScriptedPolicy([
        call("delete_workspace", {"path": path, "reason": "delete it"})
    ])

    result = ToolLoopTaskNativeUpdateAgent(policy, max_steps=1).execute(
        request(tmp_path, "math")
    )

    assert not result.finished
    assert result.trace[0]["event"] == "tool_error"
    assert (tmp_path / path).is_file()


def test_workspace_tools_reject_escape_and_symlink_delete(tmp_path) -> None:
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    (candidate / "inside.txt").write_text("public", encoding="utf-8")
    (candidate / "link.txt").symlink_to(candidate / "inside.txt")
    policy = ScriptedPolicy([
        call("glob_workspace", {"pattern": "../*"}),
        call(
            "delete_workspace",
            {"path": "link.txt", "reason": "explicitly withdrawn"},
            1,
        ),
    ])

    result = ToolLoopTaskNativeUpdateAgent(policy, max_steps=2).execute(
        request(candidate, "math")
    )

    assert not result.finished
    assert [item["event"] for item in result.trace] == ["tool_error", "tool_error"]
    assert outside.read_text(encoding="utf-8") == "secret"
    assert (candidate / "inside.txt").read_text(encoding="utf-8") == "public"


def test_glob_and_grep_results_are_bounded(tmp_path) -> None:
    (tmp_path / "answer.md").write_text("ready", encoding="utf-8")
    for index in range(105):
        (tmp_path / f"note-{index:03}.txt").write_text(
            "needle\n" * 2, encoding="utf-8"
        )
    policy = ScriptedPolicy([
        call("glob_workspace", {"pattern": "*.txt"}),
        call("grep_workspace", {"query": "needle", "pattern": "*.txt"}, 1),
        call("finish", {}, 2),
    ])

    result = ToolLoopTaskNativeUpdateAgent(policy, max_steps=3).execute(
        request(tmp_path, "math")
    )

    assert result.finished
    assert len(result.trace[0]["observation"]["paths"]) == 100
    assert result.trace[0]["observation"]["truncated"] is True
    assert len(result.trace[1]["observation"]["matches"]) == 50
    assert result.trace[1]["observation"]["truncated"] is True


def test_tool_loop_failure_does_not_commit_or_hide_typed_error(tmp_path) -> None:
    policy = ScriptedPolicy([TaskNativePolicyResponse(content="no tool")])
    result = ToolLoopTaskNativeUpdateAgent(policy, max_steps=1).execute(
        request(tmp_path, "math")
    )

    assert not result.finished
    assert result.failure is not None
    assert result.failure.kind == "protocol_error"


def test_tool_loop_rejects_path_and_parallel_tool_protocol(tmp_path) -> None:
    policy = ScriptedPolicy([
        TaskNativePolicyResponse(
            tool_calls=(
                TaskNativeToolCall("a", "read_workspace", {"path": "../secret"}),
                TaskNativeToolCall("b", "finish", {}),
            ),
        )
    ])
    result = ToolLoopTaskNativeUpdateAgent(policy, max_steps=1).execute(
        request(tmp_path, "math")
    )
    assert not result.finished
    assert result.failure is not None
    assert result.failure.kind == "protocol_error"


def test_code_wrapper_injects_only_testbed_capability(tmp_path, monkeypatch) -> None:
    observed = {}

    class Workspace(FakeTestbed):
        def __init__(self, image):
            super().__init__()
            observed["image"] = image

        def start(self):
            observed["started"] = True

        def apply_patch(self, patch):
            observed["patch"] = patch

        def cleanup(self):
            observed["cleaned"] = True

    class Delegate:
        def execute(self, delegated):
            observed["keys"] = set(delegated.public_runtime_assets)
            return ToolLoopTaskNativeUpdateAgent(
                ScriptedPolicy([call("finish", {})]), max_steps=1
            ).execute(delegated)

    monkeypatch.setattr(
        "githarness.integrations.stackplanner.swe.DockerSWEWorkspace", Workspace
    )
    original = request(tmp_path, "code")
    result = CodeTestbedTaskNativeUpdateAgent(
        image="public-image", delegate=Delegate()
    ).execute(original)

    assert result.finished
    assert observed == {
        "image": "public-image",
        "started": True,
        "keys": {"capabilities", "testbed"},
        "cleaned": True,
    }


def test_math_calculator_ports_bounded_expression_contract() -> None:
    assert calculate_expression("2 ** 5 + 7 % 4") == "35"
    with pytest.raises(ValueError, match="unsafe exponent"):
        calculate_expression("2 ** 100")
    with pytest.raises(ValueError, match="calculator accepts"):
        calculate_expression("__import__('os')")


@pytest.mark.parametrize(
    "query",
    [
        "SELECT 1; DELETE FROM users",
        "WITH changed AS (UPDATE users SET name = 'x') SELECT 1",
        "SELECT 1 -- hidden second statement",
    ],
)
def test_sql_rejects_multi_statement_write_and_comment_forms(query: str) -> None:
    with pytest.raises(ValueError):
        validate_read_only_sql(query)
    assert validate_read_only_sql("SELECT 'update' AS word;") == "SELECT 'update' AS word"


def test_sql_rejects_model_visible_database_capability(tmp_path) -> None:
    assert "read_only_sql" not in {
        tool["function"]["name"] for tool in task_native_tool_schemas("sql")
    }
    with pytest.raises(ValueError, match="unsupported sql runtime assets"):
        request(tmp_path, "sql", {"read_only_sql": object()})

    policy = ScriptedPolicy([call("read_only_sql", {"query": "SELECT 1"})])
    result = ToolLoopTaskNativeUpdateAgent(policy, max_steps=1).execute(
        request(tmp_path, "sql")
    )
    assert not result.finished
    assert result.trace == ({
        "step": 0,
        "event": "tool_error",
        "tool": "read_only_sql",
        "code": "invalid_tool_call",
    },)


def test_search_empty_result_does_not_replay_old_receipts(tmp_path) -> None:
    class TwoSearches:
        def __init__(self):
            self.calls = 0

        def search(self, query):
            self.calls += 1
            if self.calls == 1:
                return [{"docid": "d1", "rank": 1, "score": 1.0, "snippet": query}]
            return []

    policy = ScriptedPolicy([
        call("search", {"query": "first"}),
        call("search", {"query": "second"}, 1),
        call("write_artifact", {"content": json.dumps({"exact_answer": "ok"})}, 2),
        call("finish", {}, 3),
    ])
    result = ToolLoopTaskNativeUpdateAgent(policy, max_steps=4).execute(
        request(tmp_path, "search", {"search": TwoSearches()})
    )

    assert result.finished
    assert result.trace[1]["observation"]["hits"] == []
    evidence = [json.loads(line) for line in result.modified_work_state.files["evidence.jsonl"].splitlines()]
    assert [item["docid"] for item in evidence] == ["d1"]


def test_search_query_budget_is_runtime_enforced(tmp_path) -> None:
    backend = Search()
    policy = ScriptedPolicy([
        call("search", {"query": f"query-{index}"}, index)
        for index in range(5)
    ])
    result = ToolLoopTaskNativeUpdateAgent(policy, max_steps=5).execute(
        request(tmp_path, "search", {"search": backend})
    )

    assert not result.finished
    assert result.failure is not None
    assert result.trace[-1]["event"] == "tool_error"
    assert result.modified_work_state.metadata["search_query_count"] == 4


def test_research_builds_runtime_owned_citation_ledger(tmp_path) -> None:
    policy = ScriptedPolicy([
        call("search", {"query": "public topic"}),
        call("write_artifact", {"content": "# Report\nEvidence【1】"}, 1),
        call("finish", {}, 2),
    ])
    result = ToolLoopTaskNativeUpdateAgent(policy, max_steps=3).execute(
        request(tmp_path, "research", {"search": ResearchSearch()})
    )

    assert result.finished
    references = json.loads(result.modified_work_state.files["references.json"])
    assert references["1"]["url"] == "https://example.invalid/source"
    assert "research_notes.jsonl" in result.modified_work_state.files
    receipt = get_domain_spec("research").finalizer(
        tmp_path, ExecutionResult(True, result.artifact)
    )
    assert receipt["passed"] is True
    assert receipt["citations"] == 1


def test_code_test_receipt_is_runtime_owned_and_patch_bound(tmp_path) -> None:
    testbed = FakeTestbed()
    policy = ScriptedPolicy([
        call("testbed_execute", {"command": "python -m pytest -q"}),
        call("finish", {}, 1),
    ])
    result = ToolLoopTaskNativeUpdateAgent(policy, max_steps=2).execute(
        request(tmp_path, "code", {"testbed": testbed})
    )

    receipt = json.loads(result.modified_work_state.files["test_receipt.json"])
    assert receipt["workspace"] == "/testbed"
    assert receipt["test_runs"][0]["command"] == "python -m pytest -q"
    assert receipt["patch_sha256"] == hashlib.sha256(
        result.artifact.encode("utf-8")
    ).hexdigest()
    finalized = get_domain_spec("code").finalizer(
        tmp_path, ExecutionResult(True, result.artifact)
    )
    assert finalized["passed"] is True
    assert finalized["test_runs"] == 1


@pytest.mark.parametrize("domain", ["search", "research", "code"])
def test_capability_private_fields_never_enter_trace_or_work_state(
    tmp_path, domain: str
) -> None:
    canary = "PRIVATE_EVALUATOR_CANARY"

    class LeakySearch:
        def search(self, query):
            return [{
                "docid": "d1",
                "rank": 1,
                "score": 1.0,
                "snippet": "public",
                "hidden_tests": canary,
            }]

    class LeakyResearch:
        def search(self, query):
            return [{
                "title": "Public source",
                "url": "https://example.invalid/public",
                "snippet": "public",
                "reference_report": canary,
            }]

    class LeakyTestbed(FakeTestbed):
        def execute(self, command):
            return {"returncode": 0, "output": "public", "grader_feedback": canary}

    assets = {
        "search": {"search": LeakySearch()},
        "research": {"search": LeakyResearch()},
        "code": {"testbed": LeakyTestbed()},
    }[domain]
    domain_call = {
        "search": call("search", {"query": "public"}),
        "research": call("search", {"query": "public"}),
        "code": call("testbed_execute", {"command": "pytest -q"}),
    }[domain]
    policy = ScriptedPolicy([domain_call])
    result = ToolLoopTaskNativeUpdateAgent(policy, max_steps=1).execute(
        request(tmp_path, domain, assets)
    )

    serialized = json.dumps(
        {
            "trace": result.trace,
            "files": dict(result.modified_work_state.files),
            "metadata": dict(result.modified_work_state.metadata),
        },
        ensure_ascii=False,
    )
    assert canary not in serialized
