#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$ROOT/.env" ]]; then
  set -a
  source "$ROOT/.env"
  set +a
fi
CONFIG="${CONFIG:-$ROOT/configs/examples/search_remote.yaml}"
OUTPUT="${OUTPUT:-$ROOT/.runs/search}"
: "${LLM_BASE_URL:?set LLM_BASE_URL}"
: "${LLM_API_KEY:?set LLM_API_KEY}"
: "${BROWSECOMP_SEARCH_URL:?set BROWSECOMP_SEARCH_URL}"
: "${BROWSECOMP_API_KEY:?set BROWSECOMP_API_KEY}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON_BIN:-python}" -m githarness.cli run-case --config "$CONFIG" --output "$OUTPUT" "$@"
