# Datasets and environments

This directory contains the model-visible `PublicCase` files used by
GitHarness. Private answers, databases, reference reports, hidden tests, gold
patches, and grader feedback are not model inputs and are not stored with the
public cases.

`registry.json` is the canonical index of the nine public splits. Each domain
README describes the external assets and runtime needed to generate and
evaluate that domain.

| Domain | Public split | Required runtime or evaluator assets |
|---|---|---|
| [Math](math/README.md) | 100 train, 100 validation, 100 test; English; seven turns | No external inference asset; private answers for normalized exact match |
| [SQL](sql/README.md) | 100 train, 100 validation, 100 test; English; seven turns | BIRD SQLite databases for evaluation only |
| [Search](search/README.md) | 100 test; English; seven turns | BrowseComp-Plus corpus/index and Qwen3-Embedding-8B retrieval environment |
| [Research](research/README.md) | 100 test; 50 English and 50 Chinese; seven turns | BoCha-compatible search for inference; benchmark reference/judge assets for evaluation |
| [Code](code/README.md) | 100 test cases; seven turns | Docker, official SWE-bench task images, dataset, and grader |

Validate the public registry and committed artifacts without private evaluator
files:

```bash
scripts/datasets.sh validate
```

For a formal local evaluation check, place the private records at the paths
documented by each domain and run:

```bash
scripts/datasets.sh validate-private
```

The second command verifies public/private pairing locally; it does not make
private records available to generation.
