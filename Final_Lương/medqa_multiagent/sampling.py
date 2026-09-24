"""Deterministic dev-set sampling.

This module owns turning the official MedQA-USMLE ``dev`` split into a
runtime dev sample for a given run. It deliberately has no code path that
can return anything from the official ``test`` split -- that split is
reachable exclusively through `medqa_multiagent.official_eval`, so any
tuning/debugging code that only ever imports *this* module can never
accidentally see test questions.

The core primitive, `deterministic_sample`, is intentionally generic (it
knows nothing about "dev" vs "test") so that `official_eval` can reuse it
for the official test split without this module needing to expose a
test-shaped API of its own.
"""

from __future__ import annotations

import hashlib
import random
from typing import Sequence, TypeVar

from .config import RunConfig

T = TypeVar("T")


def _derive_seed(seed: int, label: str) -> int:
    """Derive a distinct, deterministic integer seed for a (seed, label) pair.

    Sampling from two different pools (e.g. dev vs. test) under the same
    run seed uses a different label, so their selections never become
    accidentally correlated, while remaining perfectly reproducible for a
    fixed (seed, label) pair -- no wall-clock or other hidden state is
    involved.
    """
    digest = hashlib.sha256(f"{seed}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big")


def deterministic_sample(
    items: Sequence[T], size: int, seed: int, label: str
) -> list:
    """Deterministically sample `size` items from `items`.

    The result is a pure function of (the contents/order of `items`, `size`,
    `seed`, `label`) only. Calling this twice with identical arguments -- in
    the same process, in a fresh process, days apart -- always returns the
    same items in the same order.

    Raises:
        ValueError: if `size` is negative, or larger than `len(items)`.
    """
    if size < 0:
        raise ValueError("size must be >= 0")
    if size > len(items):
        raise ValueError(
            f"Cannot sample {size} items from a pool of only {len(items)}"
        )

    rng = random.Random(_derive_seed(seed, label))
    indices = list(range(len(items)))
    rng.shuffle(indices)
    chosen = sorted(indices[:size])
    return [items[i] for i in chosen]


def sample_dev_set(dev_questions: Sequence[T], config: RunConfig) -> list:
    """Deterministically sample this run's dev-set subset.

    Sample size and seed are read from `config` -- never a hardcoded
    literal -- so scaling the dev set up or down, or re-deriving a past
    run's exact dev sample, is purely a config change.

    This is the ordinary, safe-to-call-repeatedly path used during
    tuning/debugging. It has no ability to return official test-set
    questions; see `medqa_multiagent.official_eval` for that distinctly
    separate path.
    """
    return deterministic_sample(
        dev_questions, config.dev_sample_size, config.seed, label="dev"
    )
