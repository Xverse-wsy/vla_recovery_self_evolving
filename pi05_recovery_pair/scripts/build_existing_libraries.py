"""Portable CPU-only entry point; see configs/data_sources.example.json."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == '__main__':
    from recovery_pair.cli import main
    sys.argv.insert(1, 'build-libraries')
    main()
