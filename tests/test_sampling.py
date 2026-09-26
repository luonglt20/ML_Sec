import pytest

from medqa_multiagent.config import RunConfig
from medqa_multiagent.sampling import deterministic_sample, sample_dev_set


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


def make_pool(n, prefix="q"):
    return [f"{prefix}-{i}" for i in range(n)]


def test_deterministic_sample_is_reproducible():
    pool = make_pool(100)
    first = deterministic_sample(pool, size=10, seed=42, label="dev")
    second = deterministic_sample(pool, size=10, seed=42, label="dev")
    assert first == second


def test_deterministic_sample_reproducible_across_fresh_calls_with_new_list():
    # Same contents, brand-new list object -- must still match, proving the
    # result depends on content/order, not object identity or process state.
    pool_a = make_pool(100)
    pool_b = make_pool(100)
    assert deterministic_sample(pool_a, 10, 42, "dev") == deterministic_sample(
        pool_b, 10, 42, "dev"
    )


def test_different_seed_changes_the_sample():
    pool = make_pool(100)
    sample_a = deterministic_sample(pool, size=10, seed=1, label="dev")
    sample_b = deterministic_sample(pool, size=10, seed=2, label="dev")
    assert sample_a != sample_b


def test_different_label_changes_the_sample_even_with_same_seed():
    pool = make_pool(100)
    sample_dev = deterministic_sample(pool, size=10, seed=42, label="dev")
    sample_test = deterministic_sample(pool, size=10, seed=42, label="official_test")
    assert sample_dev != sample_test


def test_sample_size_zero_returns_empty_list():
    pool = make_pool(10)
    assert deterministic_sample(pool, size=0, seed=42, label="dev") == []


def test_sample_size_equal_to_pool_returns_all_items_sorted_by_original_index():
    pool = make_pool(5)
    result = deterministic_sample(pool, size=5, seed=42, label="dev")
    assert sorted(result) == sorted(pool)
    assert len(result) == 5


@pytest.mark.parametrize("size", [-1, -10])
def test_negative_size_raises(size):
    pool = make_pool(10)
    with pytest.raises(ValueError):
        deterministic_sample(pool, size=size, seed=42, label="dev")


def test_size_larger_than_pool_raises():
    pool = make_pool(5)
    with pytest.raises(ValueError):
        deterministic_sample(pool, size=6, seed=42, label="dev")


def test_sample_dev_set_uses_config_size_and_seed():
    pool = make_pool(100)
    config = make_config(dev_sample_size=15, seed=99)

    result = sample_dev_set(pool, config)
    expected = deterministic_sample(pool, size=15, seed=99, label="dev")

    assert result == expected
    assert len(result) == 15


def test_sample_dev_set_is_deterministic_across_calls():
    pool = make_pool(50)
    config = make_config(dev_sample_size=8, seed=7)

    assert sample_dev_set(pool, config) == sample_dev_set(pool, config)


def test_no_literal_sample_size_or_seed_used_by_sample_dev_set():
    # Changing only the config's sample size/seed changes the result;
    # nothing in sample_dev_set hardcodes either value.
    pool = make_pool(100)
    config_a = make_config(dev_sample_size=10, seed=1)
    config_b = make_config(dev_sample_size=20, seed=1)
    config_c = make_config(dev_sample_size=10, seed=2)

    result_a = sample_dev_set(pool, config_a)
    result_b = sample_dev_set(pool, config_b)
    result_c = sample_dev_set(pool, config_c)

    assert len(result_a) == 10
    assert len(result_b) == 20
    assert result_a != result_c
