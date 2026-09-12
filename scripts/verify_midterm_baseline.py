#!/usr/bin/env python3
"""Fail if a protected midterm source file differs from its frozen SHA-256."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "security/midterm_baseline_manifest.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    mismatches = []
    for relative, expected in manifest["sha256"].items():
        actual = sha256(ROOT / relative)
        if actual != expected:
            mismatches.append(f"{relative}: expected {expected}, got {actual}")
    if mismatches:
        print("MIDTERM BASELINE VIOLATION")
        print("\n".join(mismatches))
        return 1
    print(f"Midterm baseline verified ({len(manifest['sha256'])} protected files).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
