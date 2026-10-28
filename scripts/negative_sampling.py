"""Offline ranker-row sampling; validation and evaluation must keep the full pool."""

import numpy as np


def negative_indices(labels, ratio: int = 30, seed: int = 42) -> np.ndarray:
    """Keep every recalled positive and up to ratio negatives per positive.

    Negatives are sampled uniformly without replacement from existing rows. Returned
    int64 indices preserve source order; labels and the candidate pool are not modified.
    This changes the training class prior, so fitted scores are not purchase probabilities.
    """
    if isinstance(ratio, (bool, np.bool_)) or not isinstance(ratio, (int, np.integer)):
        raise ValueError("ratio must be a positive integer")
    if ratio <= 0:
        raise ValueError("ratio must be a positive integer")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    labels = np.asarray(labels)
    # Membership in {0, 1} also excludes NaN and infinities without coercing labels.
    if labels.ndim != 1 or not np.isin(labels, [0, 1]).all():
        raise ValueError("labels must be a nonempty 1D binary array with both classes")
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    if not positive.size or not negative.size:
        raise ValueError("labels must be a nonempty 1D binary array with both classes")
    budget = min(int(ratio) * len(positive), len(negative))
    if budget < len(negative):
        negative = np.random.default_rng(int(seed)).choice(negative, size=budget, replace=False)
    return np.sort(np.concatenate([positive, negative])).astype(np.int64, copy=False)
