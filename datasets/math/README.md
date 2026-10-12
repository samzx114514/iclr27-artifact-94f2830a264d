# Math environment

The Math release contains 100 English seven-turn training cases and 100
English seven-turn validation cases derived from GSM8K train tasks, plus 100
English seven-turn test cases derived from GSM8K test tasks. The three splits
use distinct source tasks. Public cases contain evolving user requirements
only; gold answers remain evaluator-side.

## Required assets

No external model, database, retrieval corpus, or container is required to
load the public cases or run deterministic evaluation. Generation with a real
agent requires the model endpoint configured for GitHarness.

For local evaluation, provide the private answer files outside Git:

```text
.local/datasets/math/train/private_eval.jsonl
.local/datasets/math/validation/private_eval.jsonl
.local/datasets/math/test/private_eval.jsonl
```

Each private record must have the same `task_id` as its public case. Private
answers must not be passed to GitHarness generation, traces, workspaces, State
0, or checkpoints.

## Files

```text
train/public_cases.jsonl   100 model-visible training cases
validation/public_cases.jsonl 100 model-visible validation cases
test/public_cases.jsonl    100 model-visible test cases
UPSTREAM_LICENSE.txt       GSM8K MIT license
```

## Validation

Validate the committed public files:

```bash
scripts/datasets.sh validate
```

After placing the private records locally, validate task-ID pairing:

```bash
scripts/datasets.sh validate-private
```

## Generation and evaluation

Run generation using only the public cases and the configured GitHarness model
runtime. After generation has ended, evaluate predictions in a separate
process:

```bash
mtagentbench evaluate \
  --predictions .runs/math/predictions.jsonl \
  --private-records .local/datasets/math/test/private_eval.jsonl \
  --output .runs/math/evaluation
```

The Math evaluator extracts the final answer and applies deterministic
normalized exact match. It consumes no LLM tokens. A finalized artifact can
still be incorrect; public finalization and benchmark correctness are separate.

## License

The public Math cases are derived from GSM8K. The upstream OpenAI MIT
license is retained in [`UPSTREAM_LICENSE.txt`](UPSTREAM_LICENSE.txt). Gold
answers and construction-only records are not included with the public cases.
