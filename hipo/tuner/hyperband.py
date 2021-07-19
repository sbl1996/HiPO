import math

import numpy as np

from hipo.tuner.base import BaseTuner
from hipo.sample import sample


def halving(configs, losses, eta):
    n = len(configs)
    k = int(n / eta)
    indices = np.argpartition(losses, k)[:k]
    return [configs[i] for i in indices]


class HyperbandTuner(BaseTuner):

    def __init__(self, max_budget, min_budget=1, eta=3):
        super().__init__()
        self.max_budget = max_budget
        self.min_budget = min_budget
        self.eta = eta

        self._n_brackets = int(math.log(max_budget / min_budget, eta)) + 1

    def sample(self):
        return sample(self._search_space)

    def sample_n(self, n):
        return [self.sample() for i in range(n)]

    def successive_halving(self, configs, budget, s):
        for i in range(s + 1):
            values = []
            for config in configs:
                trial_id = self.run_trial(config, budget)
                value = self._trials[trial_id].value
                values.append(value)
                if self._max_trials is not None and len(self._trials) >= self._max_trials:
                    return
            budget = budget * self.eta
            configs = halving(configs, values, self.eta)

    def tune(self, objective, search_space, max_trials=None):
        self._objective = objective
        self._search_space = search_space
        self.max_trials = max_trials

        max_budget, min_budget, eta = self.max_budget, self.min_budget, self.eta
        n_brackets = self._n_brackets

        for s in range(n_brackets - 1, -1, -1):
            n = int(int(n_brackets / (s + 1)) * (eta ** s))
            configs = self.sample_n(n)
            budget = max_budget * (eta ** (-s))
            # print("s: %d, n: %d, b: %f" % (s, n, budget))
            self.successive_halving(configs, budget, s)