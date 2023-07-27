def test_rng():
    import numpy as np
    from pynicamdc.share import rng

    rng = rng.Random(1)

    _shape = (4, 3)

    rng.random_reset()
    r1 = rng.random_get(_shape)
    rng.random_reset()
    r2 = rng.random_get(_shape)

    assert not np.array_equal(r1, r2)
