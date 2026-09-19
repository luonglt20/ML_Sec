import inspect

from medqa_multiagent import official_eval, sampling
from medqa_multiagent.config import RunConfig
from medqa_multiagent.official_eval import (
    sample_official_test_set,
    take_first_official_test_set,
)
from medqa_multiagent.sampling import sample_dev_set


def make_config(**overrides):
    data = {
        "model": "gpt-4o-mini",
        "temperature": 0.0,
        "dev_sample_size": 10,
        "official_test_sample_size": 5,
        "seed": 42,
        "rag_top_k": 3,
        "rag_chunk_size": 256,
        "memory_top_k": 3,
    }
    data.update(overrides)
    return RunConfig(**data)


def make_pool(n, prefix):
    return [f"{prefix}-{i}" for i in range(n)]


def test_sample_official_test_set_uses_config_size_and_seed():
    test_pool = make_pool(50, "test")
    config = make_config(official_test_sample_size=12, seed=99)

    result = sample_official_test_set(test_pool, config)

    assert len(result) == 12
    assert all(item in test_pool for item in result)


def test_sample_official_test_set_is_deterministic_across_calls():
    test_pool = make_pool(50, "test")
    config = make_config(official_test_sample_size=7, seed=123)

    first = sample_official_test_set(test_pool, config)
    second = sample_official_test_set(test_pool, config)

    assert first == second


def test_dev_and_official_test_samples_never_overlap():
    # Dev and official test pools are disjoint by construction (they come
    # from different official splits) -- verify sampling from each
    # independently never produces an overlapping question.
    dev_pool = make_pool(100, "dev")
    test_pool = make_pool(100, "test")
    config = make_config(dev_sample_size=30, official_test_sample_size=30, seed=42)

    dev_sample = sample_dev_set(dev_pool, config)
    test_sample = sample_official_test_set(test_pool, config)

    assert set(dev_sample).isdisjoint(set(test_sample))


def test_sampling_module_exposes_no_test_accessing_function():
    """Structural guarantee: `sampling` cannot silently return test questions.

    Nothing in the public surface of `medqa_multiagent.sampling` mentions
    "test" -- the official test split is reachable only via
    `medqa_multiagent.official_eval.sample_official_test_set`.
    """
    public_names = [
        name
        for name, obj in inspect.getmembers(sampling)
        if not name.startswith("_") and inspect.isfunction(obj)
    ]
    assert public_names, "expected sampling module to expose some public functions"
    for name in public_names:
        assert "test" not in name.lower(), (
            f"sampling.{name} looks like it might touch the official test "
            "split, which must only be reachable via official_eval"
        )


def test_official_eval_module_is_the_distinct_home_of_test_access():
    assert hasattr(official_eval, "sample_official_test_set")
    assert "official" in official_eval.sample_official_test_set.__name__


def test_take_first_official_test_set_preserves_source_order():
    test_pool = make_pool(100, "test")

    result = take_first_official_test_set(test_pool, 50)

    assert result == test_pool[:50]
    assert result[0] == "test-0"
    assert result[-1] == "test-49"


def test_take_first_official_test_set_validates_size():
    for invalid_size in (0, -1, 4):
        try:
            take_first_official_test_set(["test-0", "test-1"], invalid_size)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for size={invalid_size}")
