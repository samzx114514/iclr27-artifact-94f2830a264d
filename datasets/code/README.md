# Code environment

The Code release contains 100 seven-turn SWE-bench Verified-derived
test cases. Public records contain the user turns and official instance ID.
Gold patches, hidden tests, grader feedback, images, and evaluator reports are
not included.

## Required assets

Install the pinned SWE-bench 4.0.5 evaluator and obtain the corresponding dataset
and task images outside Git:

```bash
pip install -e '.[swe]'
```

Docker must be available to the runtime host. GitHarness starts an existing
official task image with networking disabled and exposes only the container's
`/testbed` workspace to the agent; it never exposes the Docker socket as a
model tool and never pulls images automatically.

## Files

```text
test/public_cases.jsonl   100 model-visible Code cases
UPSTREAM_LICENSE.txt      SWE-bench MIT license
```

## Validation

```bash
scripts/datasets.sh validate
```

## Generation and evaluation

Set an official `instance_id` in `examples/stackplanner_code/config.yaml` and
run GitHarness. After generation, invoke the official evaluator separately:

```bash
mtagentbench evaluate \
  --predictions .runs/code/predictions.jsonl \
  --private-records .local/datasets/code/test/private_eval.jsonl \
  --swebench-workspace .local/swebench-workspace \
  --swebench-dataset /path/to/materialized/SWE-bench_Verified \
  --output .runs/code/evaluation
```

The public cases are derived from SWE-bench Verified. The upstream MIT license
is retained in [`UPSTREAM_LICENSE.txt`](UPSTREAM_LICENSE.txt). External images,
tests, datasets, and grader outputs remain subject to their official upstream
terms and are not redistributed here.
