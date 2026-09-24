"""The one and only sanctioned code path for reading the official
MedQA-USMLE ``test`` split.

Nothing in `medqa_multiagent.sampling` -- the module used for ordinary
dev-set tuning/debugging -- can return test-split questions. Anyone who
needs the official test set must explicitly import *this* module and call
`sample_official_test_set`, so an "official evaluation" pass is visually
and structurally distinct, at every call site, from routine dev-set work.

Per the project's hard invariant, this path is meant to be exercised once
per reported evaluation (a small smoke test), never for iterative tuning.
"""

from __future__ import annotations

from typing import Sequence, TypeVar

from .config import RunConfig
from .sampling import deterministic_sample

T = TypeVar("T")

#: Label used to derive this split's sampling sub-seed. Distinct from the
#: "dev" label used by `medqa_multiagent.sampling.sample_dev_set`, so dev
#: and official-test sampling never become correlated even if ever (by
#: mistake) run against the same underlying pool.
OFFICIAL_TEST_SAMPLE_LABEL = "official_test"


def sample_official_test_set(
    test_questions: Sequence[T], config: RunConfig
) -> list:
    """Deterministically sample the official test-set smoke-test subset.

    This is the *only* function in this codebase that is allowed to read
    from the official MedQA-USMLE ``test`` split. Sample size and seed are
    read from `config` -- never a hardcoded literal.

    Intended usage is a single official evaluation pass; it must not be
    called repeatedly as part of tuning/debugging (that is what
    `medqa_multiagent.sampling.sample_dev_set` is for).
    """
    return deterministic_sample(
        test_questions,
        config.official_test_sample_size,
        config.seed,
        label=OFFICIAL_TEST_SAMPLE_LABEL,
    )


def take_first_official_test_set(test_questions: Sequence[T], size: int) -> list:
    """Return the first ``size`` official-test records in source-file order.

    This explicit path exists for reproducibility requests that specify the
    first N benchmark questions rather than a seeded random sample.
    """
    if size <= 0:
        raise ValueError("official test size must be a positive integer")
    if size > len(test_questions):
        raise ValueError(
            f"Cannot take {size} questions from a test pool of {len(test_questions)}"
        )
    return list(test_questions[:size])
