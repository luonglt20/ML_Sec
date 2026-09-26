import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_protected_midterm_sources_match_frozen_manifest():
    manifest = json.loads(
        (ROOT / "security/midterm_baseline_manifest.json").read_text(encoding="utf-8")
    )
    for relative, expected in manifest["sha256"].items():
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert actual == expected, relative
