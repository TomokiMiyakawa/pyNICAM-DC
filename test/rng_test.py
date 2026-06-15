"""Smoke test for deterministic RNG behaviour used to seed test inputs."""

import numpy as np


def test_numpy_rng_determinism():
    shape = (4, 3)

    # same seed -> identical draws
    a = np.random.default_rng(1).random(shape)
    b = np.random.default_rng(1).random(shape)
    assert np.array_equal(a, b)

    # different seed -> different draws
    c = np.random.default_rng(2).random(shape)
    assert not np.array_equal(a, c)
