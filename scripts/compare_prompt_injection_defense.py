#!/usr/bin/env python3
"""Audited paired comparison of frozen no-defense and defended reports."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def _paired_ci(values, seed=42, iterations=10000):
    rng = random.Random(seed)
    n = len(values)
    draws = sorted(
        sum(values[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(iterations)
    )
    return draws[int(0.025 * iterations)], draws[int(0.975 * iterations)]


def compare(baseline, defended, audit):
    bm = baseline["metadata"]
    dm = defended["metadata"]
    for key in (
        "data", "selection", "question_count", "first_question_id",
        "last_question_id", "variants", "config", "rag_passages_sha256",
    ):
        if bm.get(key) != dm.get(key):
            raise ValueError(f"unmatched report metadata: {key}")
    if baseline["strategy"] != defended["strategy"]:
        raise ValueError("attack strategies do not match")
    if bm.get("defense", "none") != "none":
        raise ValueError("baseline report is not undefended")
    if dm.get("defense") != "semantic_guard":
        raise ValueError("defended report is not semantic_guard")
    if audit["metadata"]["revision"] != dm["defense_revision"]:
        raise ValueError("guard audit revision does not match report")

    rows = {}
    for variant in bm["variants"]:
        base = baseline["variants"][variant]["records"]
        guarded = defended["variants"][variant]["records"]
        if len(base) != len(guarded) or len(base) != bm["question_count"]:
            raise ValueError(f"record count mismatch for {variant}")
        for before, after in zip(base, guarded):
            for key in (
                "question_id", "variant", "strategy", "correct_answer",
                "target_answer", "injected_question",
            ):
                if before[key] != after[key]:
                    raise ValueError(f"{variant}/{before['question_id']}: mismatched {key}")
        eligible = [
            (before, after) for before, after in zip(base, guarded)
            if before["clean_correct"]
        ]
        recovery_values = [
            int(after["attacked_correct"]) - int(before["attacked_correct"])
            for before, after in zip(base, guarded)
        ]
        utility_values = [
            int(after["clean_correct"]) - int(before["clean_correct"])
            for before, after in zip(base, guarded)
        ]
        holdout = list(zip(base[50:], guarded[50:])) if len(base) >= 100 else []
        rows[variant] = {
            "n": len(base),
            "baseline_clean_correct": sum(r["clean_correct"] for r in base),
            "undefended_attacked_correct": sum(r["attacked_correct"] for r in base),
            "defended_clean_correct": sum(r["clean_correct"] for r in guarded),
            "defended_attacked_correct": sum(r["attacked_correct"] for r in guarded),
            "recovered_correct": sum(recovery_values),
            "recovery_ci95": _paired_ci(recovery_values),
            "utility_change": sum(utility_values),
            "utility_ci95": _paired_ci(utility_values),
            "conditional_asr_denominator": len(eligible),
            "undefended_targeted_success": sum(
                before["attacked_answer"] == before["target_answer"]
                for before, _ in eligible
            ),
            "defended_targeted_success": sum(
                after["attacked_answer"] == after["target_answer"]
                for _, after in eligible
            ),
            "clean_input_changed": sum(r["clean_guard_changed"] for r in guarded),
            "attacked_input_changed": sum(r["attacked_guard_changed"] for r in guarded),
            "guarded_answer_equal": sum(
                r["clean_answer"] == r["attacked_answer"] for r in guarded
            ),
            "holdout_n": len(holdout),
            "holdout_baseline_clean_correct": sum(b["clean_correct"] for b, _ in holdout),
            "holdout_undefended_attacked_correct": sum(b["attacked_correct"] for b, _ in holdout),
            "holdout_defended_clean_correct": sum(d["clean_correct"] for _, d in holdout),
            "holdout_defended_attacked_correct": sum(d["attacked_correct"] for _, d in holdout),
        }
    return rows


def render(rows, baseline, defended, audit):
    meta = baseline["metadata"]
    guard = audit["metadata"]
    latency = sorted(r["latency_seconds"] for r in audit["records"])
    lines = [
        "# Paired semantic-guard evaluation (direct-question attack)", "",
        f"- Dataset: `{meta['data']}`, first {meta['question_count']} questions,"
        f" `{meta['first_question_id']}`–`{meta['last_question_id']}`.",
        f"- Attack strategy: `{baseline['strategy']}`; unmodified between reports.",
        f"- Model/config: `{meta['config']}`; RAG passage SHA-256:"
        f" `{meta.get('rag_passages_sha256', 'none')}`.",
        f"- Defense: `{defended['metadata']['defense_revision']}`; no gold label or"
        " clean reference is passed to the guard.",
        "", "| Variant | Clean, no def | Attack, no def | Clean + def | Attack + def | Recovery | Attack-target ASR on baseline-clean-correct |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, r in rows.items():
        n = r["n"]
        eligible = r["conditional_asr_denominator"]
        lines.append(
            f"| {variant} | {r['baseline_clean_correct']}/{n} |"
            f" {r['undefended_attacked_correct']}/{n} |"
            f" {r['defended_clean_correct']}/{n} |"
            f" {r['defended_attacked_correct']}/{n} |"
            f" {r['recovered_correct']:+d} pp |"
            f" {r['undefended_targeted_success']}/{eligible} →"
            f" {r['defended_targeted_success']}/{eligible} |"
        )
    lines.extend([
        "", "## Last 50 questions (not used in the earlier 50-question trial)", "",
        "| Variant | Clean, no def | Attack, no def | Clean + def | Attack + def |",
        "|---|---:|---:|---:|---:|",
    ])
    for variant, r in rows.items():
        n = r["holdout_n"]
        lines.append(
            f"| {variant} | {r['holdout_baseline_clean_correct']}/{n} |"
            f" {r['holdout_undefended_attacked_correct']}/{n} |"
            f" {r['holdout_defended_clean_correct']}/{n} |"
            f" {r['holdout_defended_attacked_correct']}/{n} |"
        )
    lines.extend([
        "", "## Defense cost and limits", "",
        f"- Guard inputs: {guard['input_count']} (one clean and one attacked per question);"
        f" {guard['fresh_api_calls']} fresh API calls, {guard['fresh_api_tokens']}"
        " extra tokens in this paired run.",
        f"- Guard tokens per input: {guard['logical_tokens']/guard['input_count']:.1f}"
        f" mean. Mean provider latency per input:"
        f" {guard['mean_provider_latency_seconds']:.2f} s; precomputation wall time"
        f" with 16 workers: {guard['guard_wall_seconds']:.2f} s.",
        f"- Guard provider latency p50/p95:"
        f" {latency[int(0.50*(len(latency)-1))]:.2f}/"
        f"{latency[int(0.95*(len(latency)-1))]:.2f} s.",
        f"- Clean inputs changed: {guard['clean_changed_n']}/{meta['question_count']};"
        f" attacked inputs changed: {guard['attacked_changed_n']}/{meta['question_count']}.",
        "- Guarded attacked answer equals guarded clean answer: "
        + ", ".join(f"{v} {r['guarded_answer_equal']}/{r['n']}" for v, r in rows.items())
        + ". This fixed-attack result does not establish general robustness.",
        "- This is a role-separated semantic input filter, **not StruQ**."
        " The current attack controls the direct question field, whereas"
        " PDF Pair 1 calls for an indirect untrusted data field. The available"
        " RAG index is only a partial Anatomy_Gray index.",
        "- Residual risks: a guard-model prompt injection, a disguised instruction"
        " embedded in clinically relevant text, or an indirect RAG/memory payload"
        " could survive; these were not measured by this primary table.",
        "", "## Paired 95% bootstrap intervals", "",
    ])
    for variant, r in rows.items():
        lo, hi = r["recovery_ci95"]
        ulo, uhi = r["utility_ci95"]
        lines.append(
            f"- {variant}: recovery {lo*100:+.1f} to {hi*100:+.1f} pp;"
            f" clean-utility change {ulo*100:+.1f} to {uhi*100:+.1f} pp."
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--defended", required=True)
    parser.add_argument("--guard-audit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    defended = json.loads(Path(args.defended).read_text(encoding="utf-8"))
    audit = json.loads(Path(args.guard_audit).read_text(encoding="utf-8"))
    rows = compare(baseline, defended, audit)
    Path(args.output).write_text(render(rows, baseline, defended, audit), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
