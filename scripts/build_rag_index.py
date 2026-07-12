#!/usr/bin/env python3
"""One-time script: build the RAG retrieval index for V1 (and later,
RAG-retaining variants V2-V4).

Mirrors `download_medqa.py`'s role: a one-time, network- and (here also)
compute-heavy setup step that is NOT part of the installable
`medqa_multiagent` package and is never imported or run by the pipeline
itself. Once its output exists on disk, nothing at pipeline run-time
touches the network or re-does this work -- `rag.client_factory.build_retriever`
just loads the already-built index.

What it does, in order:
    1. Downloads the MedQA "textbooks" corpus (the `MedRAG/textbooks`
       Hugging Face dataset -- 18 medical textbooks, the canonical
       retrieval corpus paired with this benchmark in the literature; see
       DESIGN.md SS6) and writes each book's full text to one `.txt` file
       under `--corpus-dir` (default `data/rag_corpus/`), grouping the
       dataset's own (already fine-grained) rows back into whole-book
       documents so that *this project's* chunk size (`--config`'s
       `rag_chunk_size`, ~256 tokens by convention) -- not the dataset's
       own row granularity -- controls what actually gets embedded/indexed.
    2. Chunks each book's text into fixed-size passages
       (`rag.chunking.chunk_documents`) and writes their metadata
       (`rag.corpus.write_chunks`) to `--index-dir/passages.jsonl`.
    3. Embeds every passage with MedCPT's article encoder
       (`rag.embeddings.MedCptEmbeddingClient`, batched) and builds a FAISS
       flat (exact) index (`rag.index.FaissFlatIndex`), saved to
       `--index-dir`.

Requires the `rag` extra: `pip install -e ".[rag]"` (faiss-cpu,
transformers, torch, numpy) -- these are optional, heavy dependencies not
needed anywhere else in this project.

This is a genuinely expensive one-time step: the full MedRAG/textbooks
corpus is ~125,000 dataset rows across 18 books, which chunk down to many
thousands of ~256-token passages, each requiring one MedCPT embedding call
(CPU-only, no GPU required, but correspondingly slow -- expect this to run
for a substantial amount of time on a full run with no GPU). `--max-rows`
is provided purely as a development/smoke-test convenience, to build a
small index quickly while iterating on this script or the rest of the RAG
module, and must never be used to build the index actually reported on.

Usage:
    python3 scripts/build_rag_index.py --config config.json
    python3 scripts/build_rag_index.py --config config.json --max-rows 500  # smoke test
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import urllib.request
from collections import OrderedDict
from typing import Dict, List, Optional

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from medqa_multiagent.config import RunConfig  # noqa: E402
from medqa_multiagent.rag import client_factory as rag_client_factory  # noqa: E402
from medqa_multiagent.rag.chunking import chunk_documents  # noqa: E402
from medqa_multiagent.rag.corpus import load_corpus_documents, write_chunks  # noqa: E402
from medqa_multiagent.rag.embeddings import MedCptEmbeddingClient  # noqa: E402
from medqa_multiagent.rag.index import FaissFlatIndex  # noqa: E402

DATASETS_SERVER_URL = "https://datasets-server.huggingface.co/rows"
DATASET_NAME = "MedRAG/textbooks"
PAGE_SIZE = 100

DEFAULT_CORPUS_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "rag_corpus"
DEFAULT_CONFIG_PATH = "config.example.json"
DEFAULT_EMBEDDING_BATCH_SIZE = 16


def _fetch_page(offset: int, length: int, retries: int = 5) -> dict:
    url = (
        f"{DATASETS_SERVER_URL}?dataset={DATASET_NAME}"
        f"&config=default&split=train&offset={offset}&length={length}"
    )
    last_error: Exception = RuntimeError("unreachable")
    for _ in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - retry on any transient network error
            last_error = exc
    raise last_error


def download_corpus_by_book(max_rows: Optional[int] = None) -> Dict[str, List[str]]:
    """Download `MedRAG/textbooks` and group its rows by book (`title`).

    Returns an `OrderedDict` mapping book title -> ordered list of that
    book's row contents (in dataset row order), so the caller can
    concatenate each book's rows back into one whole-book document before
    re-chunking it at this project's own chunk size.
    """
    first_page = _fetch_page(0, PAGE_SIZE)
    total = first_page["num_rows_total"]
    if max_rows is not None:
        total = min(total, max_rows)

    books: "OrderedDict[str, List[str]]" = OrderedDict()
    offset = 0
    while offset < total:
        length = min(PAGE_SIZE, total - offset)
        page = _fetch_page(offset, length)
        for item in page["rows"]:
            row = item["row"]
            books.setdefault(row["title"], []).append(row["content"])
        offset += len(page["rows"])
        print(f"  downloaded {offset}/{total} rows...")
        if not page["rows"]:
            break  # pragma: no cover - defensive, avoids an infinite loop
    return books


def write_corpus_documents(books: Dict[str, List[str]], corpus_dir: pathlib.Path) -> None:
    """Write each book's concatenated row text to `corpus_dir/{title}.txt`."""
    corpus_dir.mkdir(parents=True, exist_ok=True)
    for title, rows in books.items():
        (corpus_dir / f"{title}.txt").write_text(" ".join(rows), encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to a RunConfig JSON file (default: {DEFAULT_CONFIG_PATH}). "
        "Supplies rag_chunk_size (indexing) and rag_index_dir (output location).",
    )
    parser.add_argument(
        "--corpus-dir",
        default=str(DEFAULT_CORPUS_DIR),
        help=f"Directory to write raw per-book .txt files to (default: {DEFAULT_CORPUS_DIR}).",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=DEFAULT_EMBEDDING_BATCH_SIZE,
        help=f"Passages embedded per MedCPT call (default: {DEFAULT_EMBEDDING_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help=(
            "Development/smoke-test convenience only: cap the number of "
            "dataset rows downloaded, to build a small index quickly while "
            "iterating. Never use this for the index actually reported on."
        ),
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip re-downloading the corpus; re-chunk/re-embed/re-index "
        "whatever is already in --corpus-dir.",
    )
    args = parser.parse_args(argv)

    config = RunConfig.from_json_file(args.config)
    corpus_dir = pathlib.Path(args.corpus_dir)
    index_dir = pathlib.Path(config.rag_index_dir)

    if not args.skip_download:
        print(f"Downloading {DATASET_NAME} (max_rows={args.max_rows}) ...")
        books = download_corpus_by_book(max_rows=args.max_rows)
        print(f"  downloaded {len(books)} book(s)")
        write_corpus_documents(books, corpus_dir)
        print(f"  wrote raw corpus text to {corpus_dir}")

    print(f"Chunking corpus at ~{config.rag_chunk_size} tokens/chunk ...")
    documents = load_corpus_documents(corpus_dir)
    chunks = chunk_documents(documents, config.rag_chunk_size)
    print(f"  produced {len(chunks)} chunk(s) from {len(documents)} document(s)")

    passages_path = index_dir / rag_client_factory.PASSAGES_FILENAME
    write_chunks(chunks, passages_path)
    print(f"  wrote passage metadata to {passages_path}")

    print("Embedding passages with MedCPT (this is the slow, CPU-bound step) ...")
    embedder = MedCptEmbeddingClient()
    vectors: List[List[float]] = []
    batch_size = args.embedding_batch_size
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        vectors.extend(embedder.embed_passages([chunk.text for chunk in batch]))
        print(f"  embedded {min(start + batch_size, len(chunks))}/{len(chunks)} passages...")

    print("Building FAISS flat index ...")
    index = FaissFlatIndex.build(vectors, [chunk.chunk_id for chunk in chunks])
    index.save(index_dir)
    print(f"  wrote index to {index_dir}")

    print(f"Done. {len(chunks)} passages indexed at {index_dir}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
