import numpy as np


class Random:

    def __init__(self, prc_me):
        self.prc_me = prc_me

    def get_seedvar(self):
        from datetime import datetime
        from time import process_time

        time1 = datetime.today()
        time2 = process_time()

        return (
            (time1.year - 1970) * 32140800,
            time1.month * 2678400,
            time1.day * 86400,
            time1.hour * 60,
            time1.minute * 3600,
            time1.second * 60,
            time1.microsecond,
            int(time2 * 1e6) + self.prc_me,
        )

    def random_setup(self):
        print("\n+++ Module[random]/Category[common share]\n")
        _random_seedvar = self.get_seedvar()
        self.rng = np.random.default_rng(seed=_random_seedvar)
        print(f"*** Array size for random seed: {len(_random_seedvar)}")

    def random_reset(self):
        self.rng = np.random.default_rng(seed=self.get_seedvar())

    def random_get(self, shape):
        self.random_reset()
        return self.rng.random(shape, dtype=float)
