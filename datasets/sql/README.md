# SQL environment

The SQL release contains 100 English seven-turn training cases, 100 English
seven-turn validation cases, and 100 English seven-turn test cases derived from
BIRD. Public cases include the question, complete schema text, and permitted
evidence. Gold SQL, SQLite files, execution results, and correctness signals
remain evaluator-side.

## Required assets

Generation does not require database access. For evaluation, download the
official BIRD development databases and expose their root only to the evaluator:

```text
/path/to/BIRD/dev_databases/<db_id>/<db_id>.sqlite
```

The expected database identities are recorded in `runtime_assets.json`.

## Files

```text
train/public_cases.jsonl   100 model-visible training cases
validation/public_cases.jsonl 100 model-visible validation cases
test/public_cases.jsonl    100 model-visible test cases
runtime_assets.json        expected external database hashes
LICENSE.md                 BIRD-derived data terms and attribution
```

## Validation

```bash
scripts/datasets.sh validate
```

## Generation and evaluation

The canonical root and Update Agent generate one read-only `SELECT` or `WITH`
statement from public context. They receive no SQL execution tool. After
generation, evaluate in a separate process:

```bash
mtagentbench evaluate \
  --predictions .runs/sql/predictions.jsonl \
  --private-records .local/datasets/sql/test/private_eval.jsonl \
  --db-root /path/to/BIRD/dev_databases \
  --output .runs/sql/evaluation
```

## License

Attribution and modification terms for the redistributed BIRD-derived public
material are recorded in [`LICENSE.md`](LICENSE.md). Official databases are not
redistributed.
