import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREFIX = 'pi05_prefix4_tail15_20261111'
POLICY = 'vla_prefix4_tail15_v1'
SEGMENTS = 4
ACTION_ROWS = 480
WINDOWS = 444
TOTAL_WINDOWS = 10308
TOTAL_ACTION_ROWS = 10344


def selection():
    return json.loads((HERE / 'selection.json').read_text())


def selection_sha():
    return hashlib.sha256((HERE / 'selection.json').read_bytes()).hexdigest()
