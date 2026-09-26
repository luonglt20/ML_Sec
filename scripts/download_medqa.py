#!/usr/bin/env python3
"""One-time script: populate data/dev.jsonl and data/test.jsonl with the
official MedQA-USMLE (US, 4-option) dev/test splits.

This project's harness loads the official `dev`/`test` splits directly
(rather than carving a dev set out of `train`) -- see DESIGN.md SS4. The
source used here is the public `GBaker/MedQA-USMLE-4-options-hf` Hugging
Face dataset, which mirrors the original Jin et al. (2020) MedQA release's
`validation`/`test` splits verbatim (same question/option/answer content,
just re-packaged); its `validation` split is this project's `dev`.

This script is a one-time, network-dependent setup step -- it is NOT part
of the installable `medqa_multiagent` package and is never imported or run
by the pipeline itself. Once `data/dev.jsonl` and `data/test.jsonl` exist,
nothing at pipeline run-time touches the network to read question data.

Usage:
    python3 scripts/download_medqa.py

Output files, one JSON object per line, matching the schema expected by
`medqa_multiagent.data.load_questions`:
    {"question_id": str, "question": str, "options": {"A": str, ...}, "answer": str}
"""

from __future__ import annotations

import json
import pathlib
import sys
import urllib.request

DATASETS_SERVER_URL = "https://datasets-server.huggingface.co/rows"
DATASET_NAME = "GBaker/MedQA-USMLE-4-options-hf"
PAGE_SIZE = 100

# (HF split name, output filename) -- HF's "validation" split is this
# project's "dev" (the official MedQA dev split).
SPLITS = [
    ("validation", "dev.jsonl"),
    ("test", "test.jsonl"),
]

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"

_OPTION_LETTERS = ("A", "B", "C", "D")
_ENDING_FIELDS = ("ending0", "ending1", "ending2", "ending3")


def _fetch_page(split: str, offset: int, length: int, retries: int = 5) -> dict:
    url = (
        f"{DATASETS_SERVER_URL}?dataset={DATASET_NAME}"
        f"&config=default&split={split}&offset={offset}&length={length}"
    )
    last_error: Exception = RuntimeError("unreachable")
    for _ in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - retry on any transient network error
            last_error = exc
    raise last_error


def _convert_row(row: dict) -> dict:
    """Convert one GBaker/MedQA-USMLE-4-options-hf row to this project's schema."""
    options = {
        letter: row[field] for letter, field in zip(_OPTION_LETTERS, _ENDING_FIELDS)
    }
    answer = _OPTION_LETTERS[row["label"]]
    return {
        "question_id": row["id"],
        "question": row["sent1"],
        "options": options,
        "answer": answer,
    }


def download_split(split: str) -> list:
    first_page = _fetch_page(split, offset=0, length=PAGE_SIZE)
    total = first_page["num_rows_total"]
    rows = [item["row"] for item in first_page["rows"]]

    offset = len(rows)
    while offset < total:
        page = _fetch_page(split, offset=offset, length=PAGE_SIZE)
        page_rows = [item["row"] for item in page["rows"]]
        rows.extend(page_rows)
        offset += len(page_rows)
        if not page_rows:
            break  # pragma: no cover - defensive, avoids an infinite loop

    if len(rows) != total:
        raise RuntimeError(
            f"Expected {total} rows for split {split!r}, downloaded {len(rows)}"
        )
    return [_convert_row(row) for row in rows]


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for hf_split, filename in SPLITS:
        print(f"Downloading {DATASET_NAME}[{hf_split}] -> data/{filename} ...")
        questions = download_split(hf_split)
        out_path = DATA_DIR / filename
        with open(out_path, "w", encoding="utf-8") as fh:
            for question in questions:
                fh.write(json.dumps(question, ensure_ascii=True))
                fh.write("\n")
        print(f"  wrote {len(questions)} questions to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
