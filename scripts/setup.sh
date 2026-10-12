#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-$ROOT/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip

cd "$ROOT"
"$VENV/bin/python" -m pip install -e '.[dev,yaml,stackplanner]'
if [[ "${WITH_SWE:-0}" == "1" ]]; then
  "$VENV/bin/python" -m pip install -e '.[swe]'
fi

echo "GitHarness environment is ready: $VENV"
echo "Activate it with: source $VENV/bin/activate"
if [[ ! -f "$ROOT/.env" ]]; then
  echo "Create local provider settings with: cp .env.example .env"
fi
