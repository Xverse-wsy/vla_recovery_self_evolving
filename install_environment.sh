#!/usr/bin/env bash
# Install into the selected, already-created Python 3.12 environment.
set -euo pipefail
MODE="${1:?Usage: PI05_PYTHON=/path/to/venv/bin/python bash install_environment.sh train|eval}"
PACKAGE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PI05_PYTHON:-python}"
case "$MODE" in
    train) REQUIREMENTS="$PACKAGE/requirements.txt" ;;
    eval) REQUIREMENTS="$PACKAGE/requirements-eval.txt" ;;
    *) echo 'Expected train or eval' >&2; exit 2 ;;
esac
"$PYTHON" - <<'PY'
import platform
import sys
assert sys.version_info[:2] == (3, 12), 'Use Python 3.12'
assert platform.system() == 'Linux' and platform.machine() == 'x86_64', 'Use Linux x86_64'
assert sys.prefix != sys.base_prefix or 'conda' in sys.prefix.lower(), 'Create a dedicated venv/conda environment first'
PY
"$PYTHON" -m pip install -r "$REQUIREMENTS"
if [[ "$MODE" == train ]]; then
    # Both distributions own transformers/*. Pip's dependency installation
    # order cannot guarantee that the OpenPI implementation wins. Restore it
    # explicitly, matching the working A800 training environment.
    "$PYTHON" -m pip install --force-reinstall --no-deps rlinf-transformer-openpi==4.53.2
fi
"$PYTHON" -m pip check
echo "Installed $MODE dependencies. Follow ENVIRONMENT.md for source/resource paths and verify_environment.py."
