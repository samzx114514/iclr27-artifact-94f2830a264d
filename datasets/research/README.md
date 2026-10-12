# Research environment

The Research release contains 100 seven-turn test cases: 50 English and 50
Chinese. Public cases contain the research brief and benchmark identity only;
reference reports, judge inputs, and grader feedback remain evaluator-side.

## Required assets

Inference uses a BoCha-compatible read-only `search(query)` provider:

```bash
export BOCHA_API_KEY='...'
```

Evaluation requires the official DeepResearch Bench reference reports,
task-specific criteria, and a judge-model plugin. These private assets are not
redistributed here. GitHarness renders the official English or Chinese RACE
prompt, validates the raw four-dimensional judgment, and applies the official
weights and target/reference normalization deterministically.

## Files

```text
test/public_cases.jsonl   100 model-visible test cases
UPSTREAM_LICENSE.txt      DeepResearch Bench Apache-2.0 license
```

## Validation

```bash
scripts/datasets.sh validate
```

## Generation and evaluation

Run GitHarness with `examples/stackplanner_research/config.yaml`. Both the
canonical StackPlanner root and Task-Native Update Agent receive only the
narrow public search capability. Credentials are never serialized.

After generation, provide the benchmark judge only to evaluation:

```bash
mtagentbench evaluate \
  --predictions .runs/research/predictions.jsonl \
  --private-records .local/datasets/research/private_eval.jsonl \
  --research-judge package.module:function \
  --output .runs/research/evaluation
```

The judge plugin receives the rendered official prompt and the evaluator-only
record, and returns the raw JSON judgment (as a mapping, direct JSON text, or a
`json` code block) with `comprehensiveness`, `insight`,
`instruction_following`, and `readability` lists. It must not return a
precomputed scalar score. Each private record provides:

```text
reference.prompt
reference.reference_article
reference.language                 en or zh
reference.criteria.dimension_weight
reference.criteria.criterions
asset_bindings.benchmark_id
```

Successful Research rows use `status="scored"`, `correct=null`, and store the
task-level RACE value in `score` on the `[0, 1]` scale. The batch summary reports
`mean_race`, all four mean dimension scores, `num_completed`, and `num_errors`.
Paper tables may multiply `mean_race` by 100 for display.

This release implements RACE only. It does not report the separate DeepResearch
Bench FACT metric.

## License

The public cases are derived from DeepResearch Bench. The upstream Apache-2.0
license is retained in [`UPSTREAM_LICENSE.txt`](UPSTREAM_LICENSE.txt). The
queries were organized into seven-turn evolving-requirement trajectories;
reference reports and evaluator assets are not redistributed. Obtain those
external assets under their upstream terms. RACE source provenance and the
fixed upstream commit are recorded in
[`vendor/deep_research_bench/UPSTREAM.md`](../../vendor/deep_research_bench/UPSTREAM.md).
