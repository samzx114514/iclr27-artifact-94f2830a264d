#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"

command="${1:-validate}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "$command" in
  validate)
    exec "$PYTHON_BIN" -m mtagentbench.cli release-validate "$@"
    ;;
  validate-private)
    exec "$PYTHON_BIN" -m mtagentbench.cli release-validate --require-private "$@"
    ;;
  registry)
    exec "$PYTHON_BIN" -m mtagentbench.cli release-registry "$@"
    ;;
  smoke)
    exec "$PYTHON_BIN" -m mtagentbench.cli smoke "$@"
    ;;
  smoke-evaluators)
    exec "$PYTHON_BIN" -m mtagentbench.cli smoke-evaluators "$@"
    ;;
  prepare-search-assets)
    exec "$PYTHON_BIN" -m mtagentbench.cli prepare-search-assets "$@"
    ;;
  prepare)
    exec "$PYTHON_BIN" -m mtagentbench.cli prepare "$@"
    ;;
  *)
    echo "usage: scripts/datasets.sh {validate|validate-private|registry|smoke|smoke-evaluators|prepare-search-assets|prepare} [options]" >&2
    exit 2
    ;;
esac
