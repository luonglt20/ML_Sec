"""Thin CLI wrapper over the black-box `answer_question` entrypoint.

Two subcommands:
    answer  -- answer a single question via the black-box entrypoint
               (question + options + variant in, answer + explanation out).
    run     -- run this run's configured dev sample (loaded from a local
               question-pool file and sampled via #1's `sample_dev_set`)
               through the given variant and write a durable
               prediction/trace file.

Both subcommands build the same LLM client stack: an on-disk cache (so
identical requests never re-spend API cost) wrapped in a call logger (so
every call's model/temperature/prompt is logged), around the provider
client selected by `RunConfig.model`. For RAG-using variants (currently
V1), both subcommands also build the standard RAG retriever stack (an
on-disk-cached MedCPT embedder over a pre-built FAISS index, itself
wrapped in an on-disk retrieval cache) via
`rag.client_factory.build_retriever` -- the `--rag-index-dir`,
`--embedding-cache-dir`, and `--retrieval-cache-dir` flags override its
defaults, mirroring `--cache-dir` for the LLM client.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional, Sequence

from .client_factory import DEFAULT_CACHE_DIR, build_llm_client
from .config import RunConfig
from .data import load_questions
from .entrypoint import SUPPORTED_VARIANTS, answer_question
from .env_file import load_env_file
from .pipeline import run_dev_evaluation
from .records import write_prediction_records
from .sampling import sample_dev_set

DEFAULT_ENV_FILE = ".env"


def _build_retriever_if_needed(args: argparse.Namespace, config: RunConfig):
    """Build a `Retriever` for `args.variant` if (and only if) it needs one.

    Imported lazily so that a pure-V0 CLI invocation never requires the
    `rag` extra's heavy dependencies (faiss/transformers/torch).
    """
    if args.variant == "V0" or args.variant not in SUPPORTED_VARIANTS:
        return None

    from .rag.client_factory import (
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


def cmd_answer(args: argparse.Namespace) -> int:
    config = RunConfig.from_json_file(args.config)
    client = build_llm_client(config, args.cache_dir)
    options = json.loads(args.options)
    retriever = _build_retriever_if_needed(args, config)

    result = answer_question(args.question, options, args.variant, config, client, retriever)

    print(json.dumps({"answer": result.answer, "explanation": result.explanation}))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    config = RunConfig.from_json_file(args.config)
    client = build_llm_client(config, args.cache_dir)

    pool = load_questions(args.data)
    dev_sample = sample_dev_set(pool, config)

    # Built once (rather than left for `answer_question` to lazily build
    # per-question) so a multi-question `run` doesn't reload the FAISS
    # index/MedCPT model on every iteration. V0 needs no retriever at all.
    retriever = _build_retriever_if_needed(args, config)

    records = run_dev_evaluation(dev_sample, args.variant, config, client, retriever)
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
    _add_rag_cache_arguments(answer_parser)
    answer_parser.add_argument(
        "--env-file",
        default=DEFAULT_ENV_FILE,
        help=(
            "Path to a .env file of KEY=VALUE provider API keys "
            f"(default: {DEFAULT_ENV_FILE}). Loaded before checking for "
            "required API key environment variables; values already "
            "exported in the shell always take precedence."
        ),
    )
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
    _add_rag_cache_arguments(run_parser)
    run_parser.add_argument(
        "--env-file",
        default=DEFAULT_ENV_FILE,
        help=(
            "Path to a .env file of KEY=VALUE provider API keys "
            f"(default: {DEFAULT_ENV_FILE}). Loaded before checking for "
            "required API key environment variables; values already "
            "exported in the shell always take precedence."
        ),
    )
    run_parser.set_defaults(func=cmd_run)

    return parser


def _add_rag_cache_arguments(subparser: argparse.ArgumentParser) -> None:
    """Shared RAG-related flags for both `answer` and `run`.

    All default to `None`, meaning "use `RunConfig.rag_index_dir` /
    `rag.client_factory`'s own defaults" -- only relevant for RAG-using
    variants (currently V1); ignored entirely for V0.
    """
    subparser.add_argument(
        "--rag-index-dir",
        default=None,
        help="Override RunConfig.rag_index_dir (the pre-built FAISS index/passage "
        "metadata directory from scripts/build_rag_index.py).",
    )
    subparser.add_argument(
        "--embedding-cache-dir",
        default=None,
        help="Override the on-disk embedding-call cache directory "
        "(default: rag.client_factory.DEFAULT_EMBEDDING_CACHE_DIR).",
    )
    subparser.add_argument(
        "--retrieval-cache-dir",
        default=None,
        help="Override the on-disk retrieval-call cache directory "
        "(default: rag.client_factory.DEFAULT_RETRIEVAL_CACHE_DIR).",
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    load_env_file(args.env_file)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
