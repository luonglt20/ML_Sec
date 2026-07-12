"""Thin CLI wrapper over the black-box `answer_question` entrypoint.

Two subcommands:
    answer  -- answer a single question via the black-box entrypoint
               (question + options + variant in, answer + explanation out).
    run     -- run this run's configured dev sample (loaded from a local
               question-pool file and sampled via #1's `sample_dev_set`)
               through V0 and write a durable prediction/trace file.

Both subcommands build the same LLM client stack: an on-disk cache (so
identical requests never re-spend API cost) wrapped in a call logger (so
every call's model/temperature/prompt is logged), around the provider
client selected by `RunConfig.model`.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional, Sequence

from .cache import OnDiskLLMCache
from .config import RunConfig
from .data import load_questions
from .entrypoint import answer_question
from .llm_client import LLMClient, LoggingLLMClient, create_llm_client
from .pipeline import run_dev_evaluation
from .records import write_prediction_records
from .sampling import sample_dev_set

DEFAULT_CACHE_DIR = ".cache/llm"


def _build_client(config: RunConfig, cache_dir: str) -> LLMClient:
    base_client = create_llm_client(config.model)
    cached_client = OnDiskLLMCache(base_client, cache_dir)
    return LoggingLLMClient(cached_client)


def cmd_answer(args: argparse.Namespace) -> int:
    config = RunConfig.from_json_file(args.config)
    client = _build_client(config, args.cache_dir)
    options = json.loads(args.options)

    result = answer_question(args.question, options, args.variant, config, client)

    print(json.dumps({"answer": result.answer, "explanation": result.explanation}))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    config = RunConfig.from_json_file(args.config)
    client = _build_client(config, args.cache_dir)

    pool = load_questions(args.data)
    dev_sample = sample_dev_set(pool, config)

    records = run_dev_evaluation(dev_sample, args.variant, config, client)
    write_prediction_records(records, args.output)

    print(f"Wrote {len(records)} prediction record(s) to {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="medqa-multiagent",
        description="Thin CLI wrapper over the medqa_multiagent black-box entrypoint.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    answer_parser = subparsers.add_parser(
        "answer", help="Answer a single question via the black-box entrypoint."
    )
    answer_parser.add_argument("--config", required=True, help="Path to a RunConfig JSON file.")
    answer_parser.add_argument("--variant", default="V0", help="Variant to run (default: V0).")
    answer_parser.add_argument("--question", required=True, help="The question stem text.")
    answer_parser.add_argument(
        "--options",
        required=True,
        help='JSON object of option letter -> option text, e.g. \'{"A": "...", "B": "..."}\'.',
    )
    answer_parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    answer_parser.set_defaults(func=cmd_answer)

    run_parser = subparsers.add_parser(
        "run",
        help="Run this run's configured dev sample and write a prediction/trace file.",
    )
    run_parser.add_argument("--config", required=True, help="Path to a RunConfig JSON file.")
    run_parser.add_argument("--variant", default="V0", help="Variant to run (default: V0).")
    run_parser.add_argument(
        "--data",
        required=True,
        help="Path to a JSONL question-pool file (e.g. data/dev.jsonl).",
    )
    run_parser.add_argument(
        "--output", required=True, help="Path to write the prediction/trace JSONL file to."
    )
    run_parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    run_parser.set_defaults(func=cmd_run)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
