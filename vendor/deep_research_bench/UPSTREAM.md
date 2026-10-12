# DeepResearch Bench RACE provenance

GitHarness uses the RACE prompt and deterministic scoring semantics from:

- repository: `https://github.com/Ayanami0730/deep_research_bench`
- commit: `f2735b5c3636759c22f9e936c0232de8bf545b67`
- license: Apache License 2.0, retained at
  [`datasets/research/UPSTREAM_LICENSE.txt`](../../datasets/research/UPSTREAM_LICENSE.txt)

Source files used for the adaptation:

| Upstream file | SHA-256 |
|---|---|
| `prompt/score_prompt_en.py` | `7a63dac448e3ade2b577bcca5e455b46849930c1c2c0c664b287abdfa9b1b813` |
| `prompt/score_prompt_zh.py` | `e8773c834d4351000243f5373ec9878703b8b557ee6c44efd0a0435d1fe26463` |
| `utils/score_calculator.py` | `faca06831b10f456098e63a523fe4055a48b7715e651f4b5624bc9f62c28399d` |

The adapted implementation is
[`src/mtagentbench/evaluators/race.py`](../../src/mtagentbench/evaluators/race.py).
The English and Chinese merged RACE prompts are preserved verbatim. The score
calculation preserves the upstream criterion-weighted dimension averages,
dimension weighting, and target-versus-reference normalization.

Local changes are limited to:

- dependency-free typed inputs and outputs;
- strict rejection of missing, duplicate, fuzzy-matched, or out-of-range judge
  fields;
- separation of private evaluator records from model-visible runtime state;
- GitHarness evaluation-result and macro-summary integration.

The upstream API client, article cleaner, dataset downloader, concurrent runner,
and FACT pipeline are not included.
