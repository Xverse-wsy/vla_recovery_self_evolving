import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREFIX = 'pi05_prefix10_tail15_20261111'
SEGMENTS = 10
ACTION_ROWS = 845
WINDOWS = 755
BASE_WINDOWS = 9864
TOTAL_WINDOWS = BASE_WINDOWS + WINDOWS
TOTAL_ACTION_ROWS = BASE_WINDOWS + ACTION_ROWS


def selection():
    return json.loads((HERE / 'selection.json').read_text())


def selection_sha():
    return hashlib.sha256((HERE / 'selection.json').read_bytes()).hexdigest()


def records(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
