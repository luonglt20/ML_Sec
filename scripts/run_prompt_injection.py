#!/usr/bin/env python3
"""Run the Open-Prompt-Injection-style benchmark against MedQA variants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from medqa_multiagent.client_factory import (
    DEFAULT_CACHE_DIR,
    build_llm_client,
    build_uncached_llm_client,
)
from medqa_multiagent.config import RunConfig
from medqa_multiagent.data import load_questions
from medqa_multiagent.env_file import load_env_file
from medqa_multiagent.entrypoint import SUPPORTED_VARIANTS
from medqa_multiagent.llm_client import ConcurrencyLimitedLLMClient
from medqa_multiagent.official_eval import take_first_official_test_set
from medqa_multiagent.prompt_injection import (
    ATTACK_STRATEGIES,
    AttackMetrics,
    AttackRecord,
    evaluate_prompt_injection,
    write_attack_report,
    write_benchmark_summary,
    write_multi_variant_attack_report,
)
from medqa_multiagent.sampling import sample_dev_set


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
        "--input-guard",
        action="store_true",
        help=(
            "Remove the benchmark's appended prompt-injection instruction before "
            "calling V0-V4. This is an input-level heuristic guard and disables "
            "the LLM response cache for an independent attacked evaluation."
        ),
    )
    parser.add_argument(
        "--uncached",
        action="store_true",
        help="Bypass the LLM response cache for an independent benchmark run.",
    )
    parser.add_argument("--output", default="results/prompt_injection.json")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume an interrupted --variant all run from completed variant-level "
            "checkpoints in --output."
        ),
    )
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
        default=8,
        metavar="N",
        help="Concurrent paired question evaluations per variant (default: 8).",
    )
    parser.add_argument(
        "--provider-concurrency",
        type=int,
        default=6,
        metavar="N",
        help=(
            "Maximum simultaneous LLM provider requests across workers (default: 6). "
            "Keep this below the provider's reported concurrency limit."
        ),
    )
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
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


def _load_completed_variants(
    output: Path,
    *,
    strategy: str,
    metadata: dict,
) -> dict:
    """Load whole-variant checkpoints written by an interrupted all-variant run."""
    if not output.is_file():
        raise SystemExit(f"cannot resume: checkpoint file not found: {output}")
    try:
        payload = json.loads(output.read_text(encoding="utf-8"))
        existing_metadata = payload["metadata"]
        raw_variants = payload["variants"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise SystemExit(f"cannot resume: invalid checkpoint file: {output}") from exc

    if payload.get("strategy") != strategy:
        raise SystemExit("cannot resume: --strategy differs from the checkpoint")
    # Execution tuning may change after a rate-limit failure, but the dataset,
    # sample selection, model config, and defense mode must remain identical.
    for key in (
        "data",
        "selection",
        "question_count",
        "first_question_id",
        "last_question_id",
        "config",
        "input_guard",
    ):
        if existing_metadata.get(key) != metadata[key]:
            raise SystemExit(
                f"cannot resume: checkpoint {key}={existing_metadata.get(key)!r} "
                f"does not match requested {metadata[key]!r}"
            )

    completed = {}
    for variant, value in raw_variants.items():
        try:
            records = [AttackRecord(**record) for record in value["records"]]
            metrics = AttackMetrics(**value["metrics"])
        except (KeyError, TypeError) as exc:
            raise SystemExit(f"cannot resume: malformed {variant} checkpoint") from exc
        if len(records) != metadata["question_count"] or metrics.total != len(records):
            raise SystemExit(
                f"cannot resume: {variant} is partial; choose a new --output "
                "and rerun that variant"
            )
        completed[variant] = (records, metrics)
    return completed


def main() -> int:
    args = build_parser().parse_args()
    if args.resume and args.variant != "all":
        raise SystemExit("--resume is currently supported only with --variant all")
    load_env_file(args.env_file)
    config = RunConfig.from_json_file(args.config)
    if args.progress_every <= 0:
        raise SystemExit("--progress-every must be a positive integer")
    if args.workers <= 0:
        raise SystemExit("--workers must be a positive integer")
    if args.provider_concurrency <= 0:
        raise SystemExit("--provider-concurrency must be a positive integer")
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
        "input_guard": "heuristic_suffix_guard" if args.input_guard else "none",
        "llm_cache": not (args.input_guard or args.uncached),
        "workers": args.workers,
        "provider_concurrency": args.provider_concurrency,
        "config": args.config,
    }
    print(
        f"[Prompt Injection] Dataset={args.data}, selection={selection}, "
        f"range={sample[0].question_id}..{sample[-1].question_id}",
        flush=True,
    )
    # A guard can make an attacked prompt identical to the clean prompt.  Do
    # not let the regular cache turn that into a reused response: every
    # condition must be measured as a distinct model invocation.
    base_client = (
        build_uncached_llm_client(config)
        if args.input_guard or args.uncached
        else build_llm_client(config, args.cache_dir)
    )
    client = ConcurrencyLimitedLLMClient(base_client, args.provider_concurrency)
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
        results = (
            _load_completed_variants(
                Path(args.output), strategy=args.strategy, metadata=metadata
            )
            if args.resume
            else {}
        )
        for variant in SUPPORTED_VARIANTS:
            if variant in results:
                print(f"[Prompt Injection] Resuming: {variant} is already complete; skipping.")
                continue
            print(f"[Prompt Injection] Running {variant} ...", flush=True)
            results[variant] = evaluate_prompt_injection(
                sample,
                variant,
                args.strategy,
                config,
                client,
                retriever,
                progress_callback=show_progress,
                use_input_guard=args.input_guard,
                max_workers=args.workers,
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
        use_input_guard=args.input_guard,
        max_workers=args.workers,
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
