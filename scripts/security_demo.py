#!/usr/bin/env python3
"""Render a clean -> attacked -> defended case from saved prediction JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def _load(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        record = json.loads(raw)
        if "error" not in record:
            records.append(record)
    return records


def _find_demo(
    records: Sequence[Dict[str, Any]], question_id: Optional[str], surface: str = "rag"
) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    ids = [question_id] if question_id else sorted({row["question_id"] for row in records})
    for candidate_id in ids:
        rows = [row for row in records if row["question_id"] == candidate_id]
        # Prefer the actual API comparison when it is present; retain the local
        # StruQ triplet for its separate matched-checkpoint experiment.
        for undefended_track, defended_track in (
            ("api", "api_heuristic_guard"),
            ("local_undefended", "local_struq"),
        ):
            clean = next(
                (
                    row
                    for row in rows
                    if row["model_track"] == undefended_track
                    and row["attack_family"] == "clean"
                    and row["attack_surface"] == surface
                ),
                None,
            )
            attacked = next(
                (
                    row
                    for row in rows
                    if row["model_track"] == undefended_track
                    and row["attack_family"] == "combined"
                    and row["attack_surface"] == surface
                ),
                None,
            )
            defended = next(
                (
                    row
                    for row in rows
                    if row["model_track"] == defended_track
                    and row["attack_family"] == "combined"
                    and row["attack_surface"] == surface
                ),
                None,
            )
            if clean and attacked and defended:
                return clean, attacked, defended
    raise ValueError(
        f"no question has clean, Combined-attack, and defended records for {surface}"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--question-id")
    parser.add_argument("--surface", choices=("rag", "memory"), default="rag")
    args = parser.parse_args(argv)
    clean, attacked, defended = _find_demo(
        _load(args.predictions), args.question_id, args.surface
    )
    print(f"Question ID: {clean['question_id']}")
    print(f"Gold answer: {clean['gold_answer']}")
    print(f"Attacker target: {attacked['target_option']}")
    for title, row in (
        ("1. CLEAN / UNDEFENDED", clean),
        ("2. COMBINED ATTACK / UNDEFENDED", attacked),
        (f"3. COMBINED ATTACK / {defended['model_track'].upper()}", defended),
    ):
        print(f"\n{title}")
        print(f"Prediction: {row['predicted_answer']} | correct={row['is_correct']}")
        print(row.get("raw_response", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
