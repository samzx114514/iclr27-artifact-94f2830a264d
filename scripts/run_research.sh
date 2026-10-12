#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$ROOT/.env" ]]; then
  set -a
  source "$ROOT/.env"
  set +a
fi
CONFIG="${CONFIG:-$ROOT/examples/stackplanner_research/config.yaml}"
OUTPUT="${OUTPUT:-$ROOT/.runs/research}"
: "${LLM_BASE_URL:?set LLM_BASE_URL}"
: "${LLM_API_KEY:?set LLM_API_KEY}"
: "${BOCHA_API_KEY:?set BOCHA_API_KEY}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON_BIN:-python}" -m githarness.cli run-case --config "$CONFIG" --output "$OUTPUT" "$@"
