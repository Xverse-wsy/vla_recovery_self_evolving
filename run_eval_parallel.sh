#!/usr/bin/env bash
set -euo pipefail
: "${PI05_ROOT:?Set the project root}"
: "${PI05_EVAL_PYTHON:?Set the evaluation Python}"
PACKAGE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$PI05_EVAL_PYTHON" -Bu "$PACKAGE/evaluate_parallel.py" "$@"
