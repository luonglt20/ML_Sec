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
from dataclasses import asdict, dataclass, fields
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
            retrieval corpus.
        memory_top_k: Number of case-memory exemplars retrieved per question
            for variants that use long-term memory (V3/V4).
    """

    model: str
    temperature: float
    dev_sample_size: int
    official_test_sample_size: int
    seed: int
    rag_top_k: int
    rag_chunk_size: int
    memory_top_k: int

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
        if self.memory_top_k <= 0:
            raise ValueError("memory_top_k must be a positive integer")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "RunConfig":
        """Build a `RunConfig` from a plain mapping (e.g. parsed JSON/YAML).

        Rejects both unknown and missing fields so a config file typo fails
        loudly at load time rather than silently falling back to a
        hardcoded default.
        """
        known_fields = {f.name for f in fields(cls)}
        given_fields = set(data)

        unknown = given_fields - known_fields
        if unknown:
            raise ValueError(
                f"Unknown run configuration field(s): {sorted(unknown)}"
            )

        missing = known_fields - given_fields
        if missing:
            raise ValueError(
                f"Missing run configuration field(s): {sorted(missing)}"
            )

        return cls(**{name: data[name] for name in known_fields})

    @classmethod
    def from_json_file(cls, path: Union[str, Path]) -> "RunConfig":
        """Load a `RunConfig` from a JSON file on disk."""
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return cls.from_mapping(data)

    def to_dict(self) -> dict:
        """Serialize to a plain dict, e.g. for logging alongside a run's predictions."""
        return asdict(self)
