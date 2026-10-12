# Search environment

The Search release contains 100 English seven-turn BrowseComp-Plus-derived test
cases. Public cases contain queries only; exact answers, corpus documents, and
retrieval internals are not model-visible dataset fields.

## Required assets

Paper-protocol retrieval requires the pinned external assets described in
`retrieval_assets.json`:

- `Tevatron/browsecomp-plus-corpus`;
- `Tevatron/browsecomp-plus-indexes`;
- `Qwen/Qwen3-Embedding-8B`;
- a FAISS-compatible retrieval service exposing the narrow `search(query)`
  interface.

Download or verify the assets outside Git:

```bash
scripts/datasets.sh prepare-search-assets --download --hash-files
```

Configure the retrieval service through environment variables; never commit a
real endpoint credential:

```bash
export BROWSECOMP_SEARCH_URL='https://your-retriever.example/v1'
export BROWSECOMP_API_KEY='...'
```

## Files

```text
test/public_cases.jsonl   100 model-visible test cases
retrieval_assets.json     pinned external corpus, index, and embedding model
UPSTREAM_LICENSE.txt      upstream MIT notice
```

## Validation

```bash
scripts/datasets.sh validate
```

## Generation and evaluation

Use `configs/examples/search_remote.yaml` for the paper-style remote retrieval
boundary. `examples/search/config.yaml` and `examples/search/corpus.jsonl` are a
small local smoke fixture, not the paper retrieval environment.

After generation, evaluate exact answers separately:

```bash
mtagentbench evaluate \
  --predictions .runs/search/predictions.jsonl \
  --private-records .local/datasets/search/test/private_eval.jsonl \
  --output .runs/search/evaluation
```

This release evaluates the submitted `exact_answer` with normalized exact
match (normalized EM). It does not require an LLM judge. Retrieval content and
private answers never enter State 0, workspaces, traces, or checkpoints.

## License

The retained upstream MIT notice is in
[`UPSTREAM_LICENSE.txt`](UPSTREAM_LICENSE.txt). External corpus, index, and
model assets remain governed by their upstream terms.
