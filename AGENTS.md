# Repository instructions

## Scope

GitHarness implements Git-style requirement resolution and versioned execution
for evolving user requirements. Keep the public method component names
consistent: Git Agent, canonical StackPlanner root, State 0, Task-Native Update
Agent, Branch Execution, Unified Version Interface, and Version Graph.

The repository supports a quick standalone inference workflow. Training and
large-scale rollout experiments can be run by cloning the companion Harness-RL
repository alongside GitHarness. Maintain the compatibility exports, traces,
checkpoint bindings, and evaluator results consumed by that
workflow.

The canonical StackPlanner 1 snapshot lives in `vendor/stackplanner`. Preserve
all required third-party licenses when changing vendored code or public
datasets.

## Runtime boundaries

- The Git Agent resolves requirements and selects versions; it does not solve
  the domain task or read private evaluator information.
- The canonical StackPlanner root is the only producer of initial State 0.
- The Task-Native Update Agent modifies only an isolated candidate workspace.
- A candidate becomes an accepted checkpoint only after finalization succeeds.
- ExactReuse must make zero Router, root, and Update Agent calls.
- Search exposes only `search(query)`; Code exposes only the isolated
  `/testbed` workspace.
- Answers, gold SQL, reference reports, gold patches, hidden tests, grader
  feedback, private paths, credentials, and private endpoints must never enter
  runtime state, traces, workspaces, receipts, or checkpoints.

## Repository layout

- `src/githarness`: inference contracts and runtime orchestration.
- `src/mtagentbench`: public dataset preparation, validation, and evaluation.
- `vendor/stackplanner`: canonical StackPlanner graph implementation.
- `datasets`: model-visible public cases and one public SHA-256 registry.
- `schemas`: versioned public serialization contracts.
- `scripts`: thin Bash entry points; keep implementation logic in `src`.
- `tests`: deterministic and fake-backed contract tests.

Generated outputs belong under ignored `.local/` or `.runs/` directories.
Never commit `.env`, credentials, databases, benchmark references, model
checkpoints, private evaluator records, or generated workspaces.

## Validation

Before committing changes, run:

```bash
PYTHONPATH=src:. python -m pytest -q
python -m ruff check src tests
scripts/datasets.sh validate
scripts/datasets.sh smoke
git diff --check
```
