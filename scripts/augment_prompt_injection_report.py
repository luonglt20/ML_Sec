#!/usr/bin/env python3
"""Add benchmark-style clean/attacked metrics to a prompt-injection JSON report.

This is a post-processing utility: it reads saved records and makes no API
calls. Pricing matches the benchmark's existing DeepSeek cost assumption.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

INPUT_PRICE_USD_PER_MILLION = 0.59
OUTPUT_PRICE_USD_PER_MILLION = 0.79


def summarize(records: list[dict]) -> dict:
    total = len(records)

    def side(name: str) -> dict:
        results = [record[f"{name}_result"] for record in records]
        correct = sum(record[f"{name}_correct"] for record in records)
        invalid = sum(not record[f"{name}_valid"] for record in records)
        prompt = sum(result["prompt_tokens"] for result in results)
        completion = sum(result["completion_tokens"] for result in results)
        tokens = sum(result["total_tokens"] for result in results)
        latency = sum(result["latency_seconds"] for result in results)
        cost = (
            prompt / 1_000_000 * INPUT_PRICE_USD_PER_MILLION
            + completion / 1_000_000 * OUTPUT_PRICE_USD_PER_MILLION
        )
        return {
            f"{name}_correct_count": correct,
            f"{name}_accuracy": correct / total,
            f"{name}_invalid_count": invalid,
            f"{name}_invalid_rate": invalid / total,
            f"{name}_avg_prompt_tokens": prompt / total,
            f"{name}_avg_completion_tokens": completion / total,
            f"{name}_avg_total_tokens": tokens / total,
            f"{name}_avg_latency_seconds": latency / total,
            f"{name}_total_cost_usd": cost,
        }

    metrics = side("clean") | side("attacked")
    metrics["total"] = total
    metrics["accuracy_drop"] = metrics["clean_accuracy"] - metrics["attacked_accuracy"]
    metrics["pricing_input_usd_per_million"] = INPUT_PRICE_USD_PER_MILLION
    metrics["pricing_output_usd_per_million"] = OUTPUT_PRICE_USD_PER_MILLION
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    for variant, bundle in payload["variants"].items():
        bundle["metrics"].update(summarize(bundle["records"]))

    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Clean vs attacked benchmark metrics",
        "",
        "Pricing assumption: $0.59 / 1M input tokens and $0.79 / 1M output tokens.",
        "",
        "| Variant | Clean correct | Clean acc. | Clean invalid | Clean tokens/Q | Clean latency (s) | Clean cost ($) | Attacked correct | Attacked acc. | Attacked invalid | Attacked tokens/Q | Attacked latency (s) | Attacked cost ($) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, bundle in payload["variants"].items():
        m = bundle["metrics"]
        lines.append(
            f"| {variant} | {m['clean_correct_count']}/{m['total']} | "
            f"{m['clean_accuracy'] * 100:.2f}% | {m['clean_invalid_count']}/{m['total']} | "
            f"{m['clean_avg_total_tokens']:.2f} | {m['clean_avg_latency_seconds']:.3f} | "
            f"{m['clean_total_cost_usd']:.4f} | {m['attacked_correct_count']}/{m['total']} | "
            f"{m['attacked_accuracy'] * 100:.2f}% | {m['attacked_invalid_count']}/{m['total']} | "
            f"{m['attacked_avg_total_tokens']:.2f} | {m['attacked_avg_latency_seconds']:.3f} | "
            f"{m['attacked_total_cost_usd']:.4f} |"
        )
    args.markdown_output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
