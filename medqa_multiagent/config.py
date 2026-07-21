"""Run configuration for the MedQA-USMLE multi-agent evaluation harness.

Every knob that could otherwise become a hardcoded literal buried inside
pipeline logic -- model choice, temperature, dev-set sample size,
official-test-set smoke-test sample size, random seed, RAG top-k/chunk
size, and memory top-k -- lives here instead, as data. Downstream modules
take a `RunConfig` instance as a parameter rather than importing constants,
so any prior run's exact conditions can be reconstructed later from its
config alone.
"""

from __future__ import annotations

import json
from dataclasses import MISSING, asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Union


@dataclass(frozen=True)
class RunConfig:
    """A single run's configuration, captured explicitly as data.

    Attributes:
        model: Identifier of the LLM used for every agent role in this run
            (e.g. ``"gpt-4o-mini"``, ``"deepseek-chat"``). One model is used
            consistently across all agent roles and all variants within a
            given run, so architecture -- not model capability -- is the
            isolated variable under test.
        temperature: Sampling temperature passed to every LLM call. The
            project convention is 0 for every agent/role/phase, but this is
            still config, not a literal baked into call sites.
        dev_sample_size: Number of questions to sample from the official
            MedQA-USMLE ``dev`` split for development/tuning runs.
        official_test_sample_size: Number of questions to sample from the
            official MedQA-USMLE ``test`` split for the official
            "smoke test" evaluation pass. This is deliberately small and is
            NOT the full 1273-question benchmark.
        seed: Random seed governing all deterministic sampling in this run.
        rag_top_k: Number of retrieved passages returned by the RAG module
            for each query.
        rag_chunk_size: Target chunk size (in tokens) used when indexing the
            retrieval corpus (legacy flat chunking -- kept for backward compat
            with existing config files and the original ``chunk_documents``
            path in ``build_rag_index.py``).
        memory_top_k: Number of case-memory exemplars retrieved per question
            for variants that use long-term memory (V3/V4).
        rag_index_dir: Directory containing the pre-built RAG retrieval
            index (FAISS flat index + passage metadata) produced once by
            `scripts/build_rag_index.py`. Defaults to `"data/rag_index"` so
            existing config files (from before V1/RAG landed) keep working
            unchanged -- this field has a default and is therefore optional
            in `from_mapping`/`from_json_file`.
        rag_child_chunk_size: Token size of *child* chunks indexed into FAISS
            for Parent-Child RAG (smaller → higher embedding precision).
            Defaults to 64.
        rag_parent_chunk_size: Token size of *parent* chunks returned as full
            context to the LLM when a child chunk is retrieved. Should be a
            whole multiple of ``rag_child_chunk_size``. Defaults to 512.
        rag_max_passage_tokens: Maximum (approximate) tokens per passage
            included in any LLM prompt. Passages exceeding this are truncated
            with a ``[Truncated]`` marker so the LLM knows context is partial.
            Defaults to 200.
        rag_retrieval_buffer: Extra candidate passages fetched from FAISS
            beyond ``rag_top_k`` to absorb deduplication losses when multiple
            child chunks resolve to the same parent. Defaults to 2.
    """

    model: str
    temperature: float
    dev_sample_size: int
    official_test_sample_size: int
    seed: int
    rag_top_k: int
    rag_chunk_size: int
    memory_top_k: int
    rag_index_dir: str = "data/rag_index"
    rag_chunk_overlap: int = 32
    rag_child_chunk_size: int = 64
    rag_parent_chunk_size: int = 512
    rag_max_passage_tokens: int = 200
    rag_retrieval_buffer: int = 2
    rag_use_hybrid: bool = False
    rag_use_hyde: bool = False
    rag_relevance_threshold: float = 0.0
    rag_max_retrieval_loops: int = 2
    rag_enable_backtracking: bool = False
    rag_enable_debate: bool = False
    rag_use_multi_query: bool = False
    rag_use_option_boosting: bool = False
    rag_option_boost_weight: float = 0.05
    rag_dynamic_top_k: bool = False
    rag_heuristic_compression: bool = False
    rag_adaptive_routing: bool = False
    rag_use_reranker: bool = False
    rag_use_query_pruning: bool = False
    rag_use_synonym_expansion: bool = False
    rag_use_mmr: bool = False
    rag_mmr_lambda: float = 0.7


    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be a non-empty string")
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if self.dev_sample_size <= 0:
            raise ValueError("dev_sample_size must be a positive integer")
        if self.official_test_sample_size <= 0:
            raise ValueError(
                "official_test_sample_size must be a positive integer"
            )
        if self.rag_top_k <= 0:
            raise ValueError("rag_top_k must be a positive integer")
        if self.rag_chunk_size <= 0:
            raise ValueError("rag_chunk_size must be a positive integer")
        if self.rag_chunk_overlap < 0 or self.rag_chunk_overlap >= self.rag_chunk_size:
            raise ValueError(
                "rag_chunk_overlap must be a non-negative integer less than rag_chunk_size"
            )
        if self.memory_top_k <= 0:
            raise ValueError("memory_top_k must be a positive integer")
        if self.rag_child_chunk_size <= 0:
            raise ValueError("rag_child_chunk_size must be a positive integer")
        if self.rag_parent_chunk_size <= 0:
            raise ValueError("rag_parent_chunk_size must be a positive integer")
        if self.rag_parent_chunk_size < self.rag_child_chunk_size:
            raise ValueError(
                "rag_parent_chunk_size must be >= rag_child_chunk_size"
            )
        if self.rag_max_passage_tokens <= 0:
            raise ValueError("rag_max_passage_tokens must be a positive integer")
        if self.rag_retrieval_buffer < 0:
            raise ValueError("rag_retrieval_buffer must be a non-negative integer")
        if self.rag_relevance_threshold < 0:
            raise ValueError("rag_relevance_threshold must be non-negative")
        if self.rag_max_retrieval_loops <= 0:
            raise ValueError("rag_max_retrieval_loops must be a positive integer")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "RunConfig":
        """Build a `RunConfig` from a plain mapping (e.g. parsed JSON/YAML).

        Rejects unknown fields, and rejects missing fields *unless* that
        field declares a dataclass default (e.g. `rag_index_dir`) -- so a
        config file typo or a genuinely required omission still fails
        loudly at load time, while a config file written before a new,
        defaulted field was introduced keeps loading unchanged.
        """
        all_fields = {f.name: f for f in fields(cls)}
        known_fields = set(all_fields)
        given_fields = set(data)

        unknown = given_fields - known_fields
        if unknown:
            raise ValueError(
                f"Unknown run configuration field(s): {sorted(unknown)}"
            )

        missing = known_fields - given_fields
        required_missing = {
            name
            for name in missing
            if all_fields[name].default is MISSING
            and all_fields[name].default_factory is MISSING  # type: ignore[misc]
        }
        if required_missing:
            raise ValueError(
                f"Missing run configuration field(s): {sorted(required_missing)}"
            )

        return cls(**{name: data[name] for name in given_fields & known_fields})

    @classmethod
    def from_json_file(cls, path: Union[str, Path]) -> "RunConfig":
        """Load a `RunConfig` from a JSON file on disk."""
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return cls.from_mapping(data)

    def to_dict(self) -> dict:
        """Serialize to a plain dict, e.g. for logging alongside a run's predictions."""
        return asdict(self)
