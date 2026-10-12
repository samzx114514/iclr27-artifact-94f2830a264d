"""Isolated Docker workspace and StackPlanner Code adapter."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import subprocess
from typing import Any, Mapping
import uuid

from githarness.errors import InfrastructureError
from githarness.state0 import StackPlannerRootResult, State0Request
from githarness.version_interface.protocol import CandidateWorkState
from githarness.task_native_update import (
    TaskNativeUpdateAgent,
    UpdateAgentRequest,
    UpdateAgentResult,
    UpdateExecutionFailure,
)
from githarness.task_native_update.runtime import candidate_work_state

from .adapter import StackPlannerTaskNativeUpdateAgent
from .root import CanonicalStackPlannerRootExecutor


def image_name_for_instance(instance_id: str) -> str:
    """Return the official SWE-bench evaluation image name for an instance."""
    docker_id = instance_id.replace("__", "_1776_")
    return f"docker.io/swebench/sweb.eval.x86_64.{docker_id}:latest".lower()


def image_exists(image: str) -> bool:
    result = subprocess.run(
        ["docker", "image", "inspect", image],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


@dataclass
class DockerSWEWorkspace:
    """One network-isolated SWE-bench image; Docker control is never an LLM tool."""

    image: str
    command_timeout: int = 120
    output_limit: int = 50_000
    container_id: str | None = None
    container_name: str | None = None

    def start(self) -> None:
        if not image_exists(self.image):
            raise InfrastructureError(f"required local SWE image is missing: {self.image}")
        self.container_name = f"githarness-swe-{uuid.uuid4().hex[:10]}"
        result = subprocess.run(
            [
                "docker", "run", "--detach", "--rm", "--pull=never",
                "--network=none", "--hostname", "swe-workspace", "--name",
                self.container_name, "--workdir", "/testbed", self.image, "sleep", "2h",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
        self.container_id = result.stdout.strip()

    def execute(self, command: str, timeout: int | None = None) -> dict[str, Any]:
        if not self.container_id:
            raise InfrastructureError("SWE workspace is not running")
        try:
            result = subprocess.run(
                ["docker", "exec", "--workdir", "/testbed", self.container_id, "bash", "-lc", command],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
                timeout=timeout or self.command_timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            output = error.stdout or ""
            if isinstance(output, bytes):
                output = output.decode("utf-8", errors="replace")
            return {"output": output[-self.output_limit :], "returncode": -1, "exception_info": "command timed out"}
        output = result.stdout
        if len(output) > self.output_limit:
            output = output[: self.output_limit] + f"\n... ({len(output) - self.output_limit} chars truncated)"
        return {"output": output, "returncode": result.returncode}

    def apply_patch(self, patch: str) -> None:
        if not patch.strip():
            return
        if not self.container_id:
            raise InfrastructureError("SWE workspace is not running")
        result = subprocess.run(
            ["docker", "exec", "--workdir", "/testbed", "--interactive", self.container_id, "git", "apply", "--binary", "-"],
            input=patch,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=max(self.command_timeout, 300),
            check=False,
        )
        if result.returncode != 0:
            raise InfrastructureError(f"base patch is incompatible with the SWE image: {result.stdout[-2000:]}")

    def git_diff(self) -> str:
        result = self.execute("git add -A && git diff --cached --binary", timeout=max(self.command_timeout, 300))
        if result["returncode"] != 0:
            raise InfrastructureError(f"failed to export git diff: {result['output']}")
        return str(result["output"])

    def cleanup(self) -> bool:
        reference = self.container_id or self.container_name
        self.container_id = None
        self.container_name = None
        if not reference:
            return True
        result = subprocess.run(
            ["docker", "rm", "--force", reference],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=60,
            check=False,
        )
        return result.returncode == 0


class CodeTaskNativeUpdateAgent:
    """Run Code turns in a fresh image, applying the accepted base patch first."""

    def __init__(
        self,
        *,
        image: str,
        vendor_root: Path | None = None,
        configurable: Mapping[str, Any] | None = None,
    ) -> None:
        self.image = image
        self.vendor_root = vendor_root
        self.configurable = dict(configurable or {})

    def execute(self, request: UpdateAgentRequest) -> UpdateAgentResult:
        if request.domain != "code":
            raise ValueError("Code Update Agent requires the code domain")
        docker_workspace = DockerSWEWorkspace(self.image)
        try:
            docker_workspace.start()
            inherited_patch = request.candidate.root / "patch.diff"
            if inherited_patch.is_file():
                docker_workspace.apply_patch(inherited_patch.read_text(encoding="utf-8"))
            agent = StackPlannerTaskNativeUpdateAgent(
                domain="code",
                vendor_root=self.vendor_root,
                configurable={**self.configurable, "swe_workspace": docker_workspace},
            )
            return agent.execute(request)
        except Exception as error:
            return UpdateAgentResult(
                False,
                "patch.diff",
                "",
                candidate_work_state(request),
                failure=UpdateExecutionFailure(
                    "infrastructure_failure",
                    f"Code /testbed capability failed ({type(error).__name__})",
                    retryable=True,
                ),
            )
        finally:
            docker_workspace.cleanup()


CodeStackPlannerUpdateAgent = CodeTaskNativeUpdateAgent


class CodeTestbedTaskNativeUpdateAgent:
    """Inject only a network-isolated /testbed capability into a tool-loop agent."""

    def __init__(self, *, image: str, delegate: TaskNativeUpdateAgent) -> None:
        self.image = image
        self.delegate = delegate

    def execute(self, request: UpdateAgentRequest) -> UpdateAgentResult:
        if request.domain != "code":
            raise ValueError("Code /testbed Update Agent requires the code domain")
        testbed = DockerSWEWorkspace(self.image)
        try:
            testbed.start()
            inherited_patch = request.candidate.root / "patch.diff"
            if inherited_patch.is_file():
                testbed.apply_patch(inherited_patch.read_text(encoding="utf-8"))
            delegated = replace(
                request,
                public_runtime_assets={
                    "capabilities": ["/testbed"],
                    "testbed": testbed,
                },
            )
            return self.delegate.execute(delegated)
        except Exception as error:
            return UpdateAgentResult(
                False,
                "patch.diff",
                "",
                candidate_work_state(request),
                failure=UpdateExecutionFailure(
                    "infrastructure_failure",
                    f"Code /testbed capability failed ({type(error).__name__})",
                    retryable=True,
                ),
            )
        finally:
            testbed.cleanup()


class CodeStackPlannerRootExecutor:
    """Create Code State 0 in a fresh network-isolated /testbed workspace."""

    domain = "code"

    def __init__(
        self,
        *,
        image: str,
        vendor_root: Path | None = None,
        configurable: Mapping[str, Any] | None = None,
    ) -> None:
        self.image = image
        self.vendor_root = vendor_root
        self.configurable = dict(configurable or {})

    def create_state0(
        self,
        candidate: CandidateWorkState,
        request: State0Request,
    ) -> StackPlannerRootResult:
        docker_workspace = DockerSWEWorkspace(self.image)
        try:
            docker_workspace.start()
            executor = CanonicalStackPlannerRootExecutor(
                domain="code",
                vendor_root=self.vendor_root,
                configurable={**self.configurable, "swe_workspace": docker_workspace},
            )
            return executor.create_state0(candidate, request)
        finally:
            docker_workspace.cleanup()
