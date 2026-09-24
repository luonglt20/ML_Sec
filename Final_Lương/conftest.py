import sys
from pathlib import Path

# Ensure tests/ directory is in sys.path for pytest and IDE resolution
tests_dir = str(Path(__file__).resolve().parent / "tests")
if tests_dir not in sys.path:
    sys.path.insert(0, tests_dir)
