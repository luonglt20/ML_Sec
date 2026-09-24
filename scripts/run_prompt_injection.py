#!/usr/bin/env python3
"""Run the Open-Prompt-Injection-style benchmark against MedQA variants."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from medqa_multiagent.cache import OnDiskLLMCache
from medqa_multiagent.client_factory import DEFAULT_CACHE_DIR, build_llm_client
from medqa_multiagent.config import RunConfig
from medqa_multiagent.data import load_questions
from medqa_multiagent.env_file import load_env_file
from medqa_multiagent.entrypoint import SUPPORTED_VARIANTS
from medqa_multiagent.official_eval import take_first_official_test_set
from medqa_multiagent.prompt_injection import (
    ATTACK_STRATEGIES,
    _ConcurrentClient,
    choose_target_answer,
    evaluate_prompt_injection,
    inject_prompt,
    write_attack_report,
    write_benchmark_summary,
    write_multi_variant_attack_report,
)
from medqa_multiagent.sampling import sample_dev_set
from medqa_multiagent.prompt_defense import DEFENSE_REVISION, guard_question
from medqa_multiagent.semantic_defense import (
    SEMANTIC_GUARD_REVISION,
    SEMANTIC_GUARD_SYSTEM,
    RoleSeparatedGuardClient,
    parse_guard_response,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Paired clean/attacked MedQA prompt-injection evaluation."
    )
    parser.add_argument("--config", default="config_deepseek.json")
    parser.add_argument("--data", default="data/dev.jsonl")
    parser.add_argument(
        "--first",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Use the first N records in file order. Intended for an explicit "
            "official-test benchmark such as --data data/test.jsonl --first 50."
        ),
    )
    parser.add_argument(
        "--variant",
        choices=("all", *SUPPORTED_VARIANTS),
        default="all",
        help="Variant to evaluate, or 'all' for V0-V4 (default: all).",
    )
    parser.add_argument("--strategy", choices=ATTACK_STRATEGIES, default="combine")
    parser.add_argument(
        "--defense", choices=("none", "line_guard", "semantic_guard"), default="none",
        help="Question-input defense: none, historical line guard, or role-separated semantic guard.",
    )
    parser.add_argument("--output", default="results/prompt_injection.json")
    parser.add_argument(
        "--summary-output",
        default=None,
        help="Markdown dashboard path (default: output path with .md suffix).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        metavar="N",
        help="Refresh live metrics every N completed questions (default: 1).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=50,
        metavar="N",
        help="Concurrent question pairs per variant (default: 50; use 1 for serial).",
    )
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--guard-cache-dir", default=".cache/semantic_guard")
    parser.add_argument("--guard-workers", type=int, default=16)
    parser.add_argument("--guard-audit-output", default=None)
    parser.add_argument("--rag-index-dir", default=None)
    parser.add_argument("--embedding-cache-dir", default=None)
    parser.add_argument("--retrieval-cache-dir", default=None)
    return parser


def _build_retriever(args: argparse.Namespace, config: RunConfig):
    if args.variant == "V0":
        return None
    from medqa_multiagent.rag.client_factory import (
        DEFAULT_EMBEDDING_CACHE_DIR,
        DEFAULT_RETRIEVAL_CACHE_DIR,
        build_retriever,
    )
    return build_retriever(
        config,
        index_dir=args.rag_index_dir,
        embedding_cache_dir=args.embedding_cache_dir or DEFAULT_EMBEDDING_CACHE_DIR,
        retrieval_cache_dir=args.retrieval_cache_dir or DEFAULT_RETRIEVAL_CACHE_DIR,
    )


def _prepare_semantic_guard(args, sample, config):
    if config.model != "deepseek-chat":
        raise SystemExit("semantic_guard currently requires deepseek-chat")
    system_hash = hashlib.sha256(SEMANTIC_GUARD_SYSTEM.encode("utf-8")).hexdigest()
    client = _ConcurrentClient(
        OnDiskLLMCache(RoleSeparatedGuardClient(), args.guard_cache_dir)
    )
    inputs = {}
    for item in sample:
        target = choose_target_answer(item.options, item.answer)
        attacked = inject_prompt(item.question, target, args.strategy)
        inputs.setdefault(item.question, []).append((item.question_id, "clean"))
        inputs.setdefault(attacked, []).append((item.question_id, "attacked"))
    started = time.monotonic()

    def run_guard(raw_text):
        response = client.complete(
            role=SEMANTIC_GUARD_REVISION,
            prompt=raw_text,
            model=config.model,
            temperature=0.0,
            retrieved_context_id=system_hash,
        )
        return parse_guard_response(raw_text, response)

    guarded = {}
    with ThreadPoolExecutor(max_workers=min(args.guard_workers, len(inputs))) as pool:
        futures = {pool.submit(run_guard, raw): raw for raw in inputs}
        for future in as_completed(futures):
            result = future.result()
            guarded[futures[future]] = result
            if len(guarded) % 10 == 0 or len(guarded) == len(inputs):
                print(f"[Semantic Guard] {len(guarded)}/{len(inputs)} inputs", flush=True)
    wall_seconds = time.monotonic() - started
    clean_changed = sum(
        guarded[item.question].question != item.question for item in sample
    )
    attacked_changed = sum(
        guarded[inject_prompt(
            item.question, choose_target_answer(item.options, item.answer), args.strategy
        )].question != inject_prompt(
            item.question, choose_target_answer(item.options, item.answer), args.strategy
        ) for item in sample
    )
    metadata = {
        "revision": SEMANTIC_GUARD_REVISION,
        "system_prompt_sha256": system_hash,
        "input_count": len(inputs),
        "fresh_api_calls": sum(not result.cache_hit for result in guarded.values()),
        "cache_hits": sum(result.cache_hit for result in guarded.values()),
        "fresh_api_tokens": sum(
            result.total_tokens for result in guarded.values() if not result.cache_hit
        ),
        "logical_tokens": sum(result.total_tokens for result in guarded.values()),
        "mean_provider_latency_seconds": sum(
            result.latency_seconds for result in guarded.values()
        ) / len(guarded),
        "guard_wall_seconds": wall_seconds,
        "clean_changed_n": clean_changed,
        "attacked_changed_n": attacked_changed,
    }
    audit_path = Path(args.guard_audit_output or Path(args.output).with_suffix(".guard.json"))
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(
            {
                "metadata": metadata,
                "records": [
                    {"sources": inputs[raw], **asdict(guarded[raw])}
                    for raw in inputs
                ],
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    return {raw: result.question for raw, result in guarded.items()}, metadata, str(audit_path)

def main() -> int:
    args = build_parser().parse_args()
    load_env_file(args.env_file)
    config = RunConfig.from_json_file(args.config)
    if args.progress_every <= 0:
        raise SystemExit("--progress-every must be a positive integer")
    if args.workers <= 0:
        raise SystemExit("--workers must be a positive integer")
    if args.guard_workers <= 0:
        raise SystemExit("--guard-workers must be a positive integer")
    pool = load_questions(args.data)
    if args.first is not None:
        sample = take_first_official_test_set(pool, args.first)
        selection = f"first_{args.first}"
    else:
        sample = sample_dev_set(pool, config)
        selection = f"seeded_dev_sample_{len(sample)}"
    metadata = {
        "data": args.data,
        "selection": selection,
        "question_count": len(sample),
        "first_question_id": sample[0].question_id,
        "last_question_id": sample[-1].question_id,
        "variants": list(SUPPORTED_VARIANTS) if args.variant == "all" else [args.variant],
        "paired_clean_and_attacked": True,
        "config": args.config,
        "workers": min(args.workers, len(sample)),
        "defense": args.defense,
    }
    if args.defense == "line_guard":
        metadata["defense_revision"] = DEFENSE_REVISION
        question_guard = guard_question
    elif args.defense == "semantic_guard":
        guard_map, guard_metadata, audit_path = _prepare_semantic_guard(args, sample, config)
        metadata["defense_revision"] = SEMANTIC_GUARD_REVISION
        metadata["guard"] = guard_metadata
        metadata["guard_audit_output"] = audit_path
        question_guard = guard_map.__getitem__
    else:
        question_guard = None
    index_dir = Path(args.rag_index_dir or config.rag_index_dir)
    passages_path = index_dir / "passages.jsonl"
    if args.variant != "V0" and passages_path.is_file():
        metadata["rag_index_dir"] = str(index_dir)
        metadata["rag_passages_sha256"] = hashlib.sha256(
            passages_path.read_bytes()
        ).hexdigest()
    print(
        f"[Prompt Injection] Dataset={args.data}, selection={selection}, "
        f"range={sample[0].question_id}..{sample[-1].question_id}, "
        f"workers={min(args.workers, len(sample))}",
        flush=True,
    )
    client = build_llm_client(config, args.cache_dir)
    retriever = _build_retriever(args, config)
    summary_output = args.summary_output or str(Path(args.output).with_suffix(".md"))

    def show_progress(completed, total, record, metrics):
        if completed % args.progress_every != 0 and completed != total:
            return
        width = 20
        filled = round(width * completed / total)
        bar = "█" * filled + "░" * (width - filled)
        print(
            f"[{record.variant}] [{bar}] {completed:>2}/{total} "
            f"clean={metrics.clean_accuracy * 100:5.1f}% "
            f"attacked={metrics.attacked_accuracy * 100:5.1f}% "
            f"drop={metrics.accuracy_drop * 100:+5.1f}pp "
            f"ASR={metrics.attack_success_rate * 100:5.1f}% "
            f"flip={metrics.prediction_flip_rate * 100:5.1f}%",
            flush=True,
        )

    if args.variant == "all":
        results = {}
        for variant in SUPPORTED_VARIANTS:
            print(f"[Prompt Injection] Running {variant} ...", flush=True)
            results[variant] = evaluate_prompt_injection(
                sample,
                variant,
                args.strategy,
                config,
                client,
                retriever,
                progress_callback=show_progress,
                max_workers=args.workers,
                question_guard=question_guard,
            )
            # Checkpoint after every variant so an API/native failure later in
            # the ladder never discards already-completed results.
            write_multi_variant_attack_report(
                results, args.strategy, args.output, metadata=metadata
            )
            write_benchmark_summary(
                results, args.strategy, summary_output, metadata=metadata
            )
            print(f"[Prompt Injection] Checkpointed {variant} to {args.output}", flush=True)
        summary = {
            variant: metrics.to_dict()
            for variant, (_, metrics) in results.items()
        }
        print(json.dumps(summary, indent=2))
        print(
            f"Wrote {len(sample)} paired record(s) per variant "
            f"for {len(results)} variants to {args.output}"
        )
        return 0

    records, metrics = evaluate_prompt_injection(
        sample,
        args.variant,
        args.strategy,
        config,
        client,
        retriever,
        progress_callback=show_progress,
        max_workers=args.workers,
        question_guard=question_guard,
    )
    write_attack_report(records, metrics, args.output, metadata=metadata)
    write_benchmark_summary(
        {args.variant: (records, metrics)},
        args.strategy,
        summary_output,
        metadata=metadata,
    )
    print(json.dumps(metrics.to_dict(), indent=2))
    print(f"Wrote {len(records)} paired record(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
