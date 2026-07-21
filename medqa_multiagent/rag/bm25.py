"""A pure Python implementation of the BM25 search algorithm.

Used for sparse retrieval alongside FAISS's dense semantic retrieval in
a Hybrid Search scheme. Calculates token-matching scores based on TF-IDF,
scaled by document length normalization (BM25 OKAPI variant). Zero-dependency,
reproducible, and fully serializable.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple, Union


def tokenize(text: str) -> List[str]:
    """Lowercase text and split into words, removing basic punctuation."""
    text = text.lower()
    # Replace non-alphanumeric with spaces to clean tokens
    words = re.findall(r"\b\w+\b", text)
    return words


class BM25Index:
    """Zero-dependency BM25 index over a collection of document chunks.

    Attributes:
        k1: Term frequency saturation parameter (default 1.5).
        b: Document length normalization parameter (default 0.75).
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.doc_ids: List[str] = []
        # doc_id -> list of tokens
        self.doc_lengths: Dict[str, int] = {}
        # token -> doc_id -> count
        self.term_frequencies: Dict[str, Dict[str, int]] = {}
        # token -> document frequency (count of docs containing token)
        self.doc_frequencies: Dict[str, int] = {}
        self.avg_doc_len: float = 0.0

    @classmethod
    def build(
        cls,
        documents: Sequence[Tuple[str, str]],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> BM25Index:
        """Build an index from a sequence of (doc_id, text) pairs."""
        index = cls(k1=k1, b=b)
        if not documents:
            return index

        total_length = 0
        for doc_id, text in documents:
            index.doc_ids.append(doc_id)
            tokens = tokenize(text)
            index.doc_lengths[doc_id] = len(tokens)
            total_length += len(tokens)

            # Calculate TF within this doc
            counts: Dict[str, int] = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1

            for token, count in counts.items():
                if token not in index.term_frequencies:
                    index.term_frequencies[token] = {}
                index.term_frequencies[token][doc_id] = count
                index.doc_frequencies[token] = index.doc_frequencies.get(token, 0) + 1

        index.avg_doc_len = total_length / len(documents) if documents else 0.0
        return index

    def idf(self, token: str) -> float:
        """Calculate the Inverse Document Frequency (IDF) of a token."""
        n = self.doc_frequencies.get(token, 0)
        N = len(self.doc_ids)
        # Standard Okapi BM25 IDF formulation with smoothing to avoid negatives
        return math.log(1 + (N - n + 0.5) / (n + 0.5))

    def search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        """Search the index using BM25 scoring.

        Returns:
            List of (doc_id, score) pairs, sorted by descending score.
        """
        query_tokens = tokenize(query)
        if not query_tokens or not self.doc_ids:
            return []

        scores: Dict[str, float] = {}

        # Calculate scores only for docs containing query tokens
        for token in set(query_tokens):
            if token not in self.term_frequencies:
                continue

            idf_val = self.idf(token)
            tfs = self.term_frequencies[token]

            for doc_id, tf in tfs.items():
                doc_len = self.doc_lengths[doc_id]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
                scores[doc_id] = scores.get(doc_id, 0.0) + idf_val * (numerator / denominator)

        # Sort and return top_k
        sorted_hits = sorted(scores.items(), key=lambda x: -x[1])
        return sorted_hits[:top_k]

    def save(self, path: Union[str, Path]) -> None:
        """Serialize the index to a JSON file."""
        data = {
            "k1": self.k1,
            "b": self.b,
            "doc_ids": self.doc_ids,
            "doc_lengths": self.doc_lengths,
            "term_frequencies": self.term_frequencies,
            "doc_frequencies": self.doc_frequencies,
            "avg_doc_len": self.avg_doc_len,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> BM25Index:
        """Load a serialized index from a JSON file."""
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        index = cls(k1=data["k1"], b=data["b"])
        index.doc_ids = data["doc_ids"]
        index.doc_lengths = data["doc_lengths"]
        index.term_frequencies = data["term_frequencies"]
        index.doc_frequencies = data["doc_frequencies"]
        index.avg_doc_len = data["avg_doc_len"]
        return index
