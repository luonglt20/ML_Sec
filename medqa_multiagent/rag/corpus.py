"""Loading raw corpus documents from disk, and persisting chunk metadata.

`load_corpus_documents` owns reading a directory of raw `.txt` documents
(one file per source textbook, as produced by
`scripts/build_rag_index.py`'s corpus-download step) into `(source, text)`
pairs, in a deterministic (sorted-by-filename) order, ready for
`chunking.chunk_documents`.

`write_chunks`/`read_chunks` persist a chunk list as JSONL, the same
passage-metadata format `rag.index.FaissFlatIndex` and `rag.retriever`
consume at query time -- decoupled from any embedding/index concern, so it
can be unit-tested with plain files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple, Union

from .chunking import Chunk

_TEXT_SUFFIX = ".txt"


def load_corpus_documents(directory: Union[str, Path]) -> List[Tuple[str, str]]:
    """Load every `*.txt` file in `directory` as one `(source, text)` pair.

    `source` is the filename without its `.txt` suffix (e.g.
    ``"Anatomy_Gray"``). Files are read in sorted-filename order, so the
    result -- and everything chunked/indexed from it -- is deterministic
    across repeated indexing runs given the same directory contents.

    Raises:
        FileNotFoundError: if `directory` doesn't exist.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Corpus directory not found: {directory}")

    documents: List[Tuple[str, str]] = []
    for path in sorted(directory.glob(f"*{_TEXT_SUFFIX}")):
        source = path.stem
        text = path.read_text(encoding="utf-8")
        documents.append((source, text))
    return documents


def write_chunks(chunks: List[Chunk], path: Union[str, Path]) -> None:
    """Write `chunks` as passage metadata, one JSON object per line.

    Creates parent directories as needed. This is the on-disk format
    `read_chunks` and `rag.index`/`rag.retriever` read back at query time.

    The optional ``parent_id`` field is written when present (Parent-Child
    RAG chunks), and omitted for plain flat chunks so existing passage files
    stay compact and backward-compatible.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for chunk in chunks:
            record: dict = {"chunk_id": chunk.chunk_id, "source": chunk.source, "text": chunk.text}
            if chunk.parent_id is not None:
                record["parent_id"] = chunk.parent_id
            fh.write(json.dumps(record, ensure_ascii=True))
            fh.write("\n")


def read_chunks(path: Union[str, Path]) -> List[Chunk]:
    """Read back a chunk-metadata JSONL file written by `write_chunks`.

    Reads the optional ``parent_id`` field when present (Parent-Child RAG
    files), and defaults it to ``None`` for flat-chunk files that predate
    the hierarchical format -- fully backward-compatible.
    """
    chunks: List[Chunk] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            chunks.append(
                Chunk(
                    chunk_id=data["chunk_id"],
                    source=data["source"],
                    text=data["text"],
                    parent_id=data.get("parent_id"),  # None for legacy files
                )
            )
    return chunks
