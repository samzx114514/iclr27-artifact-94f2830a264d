"""Typed errors used across the framework."""


class GitHarnessError(RuntimeError):
    """Base class for framework errors."""


class ProtocolError(GitHarnessError):
    """A model or caller violated the public protocol."""


class InfrastructureError(GitHarnessError):
    """An external service, workspace, or dependency failed."""


class ModelError(GitHarnessError):
    """A model request or response failed."""


class RuntimeFinalizationError(GitHarnessError):
    """A candidate failed non-semantic runtime checks."""


class State0Error(GitHarnessError):
    """State 0 creation, serialization, or acceptance failed."""


class RootExecutionError(State0Error):
    """The canonical StackPlanner root failed before State 0 acceptance."""


class State0SerializationError(State0Error):
    """A State 0 payload violated its versioned schema."""


class CheckpointError(GitHarnessError):
    """Checkpoint persistence or binding failed."""


class TaskEvaluationError(GitHarnessError):
    """A task evaluator could not produce a valid result."""


class DatasetConstructionError(GitHarnessError):
    """A benchmark source could not be projected safely."""


class DataLeakageError(DatasetConstructionError):
    """Private evaluation data appeared in a public artifact."""
