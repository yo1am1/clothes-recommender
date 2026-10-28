import numpy as np
import pytest

from scripts.negative_sampling import negative_indices


def test_default_budget_keeps_every_positive_and_source_order_without_mutation():
    labels = np.zeros(250)
    positives = np.array([1, 17, 52, 150])
    labels[positives] = 1
    original = labels.copy()
    indices = negative_indices(labels)

    assert indices.dtype == np.int64
    assert indices.ndim == 1
    assert len(indices) == 4 + 30 * 4
    assert np.all(np.diff(indices) > 0)
    assert set(positives).issubset(indices)
    assert labels[indices].sum() == 4
    assert np.count_nonzero(labels[indices] == 0) == 120
    np.testing.assert_array_equal(labels, original)
    np.testing.assert_array_equal(indices, negative_indices(labels))


def test_fixed_seed_is_deterministic_and_different_seeds_really_change_negatives():
    labels = np.array([1] * 10 + [0] * 300)
    first = negative_indices(labels, ratio=3, seed=42)
    second = negative_indices(labels, ratio=3, seed=43)
    np.testing.assert_array_equal(first, negative_indices(labels, ratio=3, seed=42))
    assert len(first) == len(second) == 40
    assert set(first) != set(second)
    assert set(range(10)).issubset(first)
    assert set(range(10)).issubset(second)


@pytest.mark.parametrize("labels", [[1, 0, 1], [True, False], [0.0, 1.0, 0.0]])
def test_short_negative_pool_keeps_all_existing_rows(labels):
    np.testing.assert_array_equal(negative_indices(labels), np.arange(len(labels), dtype=np.int64))


def test_integer_numpy_parameters_and_large_ratio_do_not_overflow():
    labels = np.array([1, 0, 0, 0])
    assert len(negative_indices(labels, ratio=np.int64(1), seed=np.uint64(0))) == 2
    np.testing.assert_array_equal(negative_indices(labels, ratio=10**100), np.arange(4))


def test_readonly_labels_and_global_random_state_are_untouched():
    labels = np.array([1] + [0] * 100)
    labels.flags.writeable = False
    before = np.random.get_state()
    negative_indices(labels, ratio=1)
    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


@pytest.mark.parametrize(
    "labels",
    [
        [],
        [0, 0],
        [1, 1],
        [[0, 1]],
        1,
        [0, 1, 2],
        [0, -1, 1],
        [0, 0.5, 1],
        [0, np.nan, 1],
        [0, np.inf, 1],
        [0, -np.inf, 1],
        ["0", "1"],
        [0, None, 1],
    ],
)
def test_invalid_labels_are_rejected(labels):
    with pytest.raises(ValueError, match="labels"):
        negative_indices(labels)


@pytest.mark.parametrize("ratio", [0, -1, 1.0, True, False, np.bool_(True), "30", None])
def test_invalid_ratio_is_rejected_even_when_pool_is_short(ratio):
    with pytest.raises(ValueError, match="ratio"):
        negative_indices([0, 1], ratio=ratio)


@pytest.mark.parametrize("seed", [-1, 0.0, True, False, np.bool_(False), "42", None])
def test_invalid_seed_is_rejected_even_when_pool_is_short(seed):
    with pytest.raises(ValueError, match="seed"):
        negative_indices([0, 1], seed=seed)
