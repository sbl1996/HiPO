import math
import numpy as np

from hipo.sample import sample


def halving(configs, losses, eta):
    n = len(configs)
    k = int(n / eta)
    indices = np.argpartition(losses, k)[:k]
    return [configs[i] for i in indices]


class HyperbandTuner:

    def __init__(self, R, eta):
        self.R = R
        self.eta = eta

        self._trials = []
        self._objective = None
        self._search_space = None

    def sample(self):
        return sample(self._search_space)

    def sample_n(self, n):
        return [self.sample() for i in range(n)]

    def run_trial(self, hparams, budget):
        metric = self._objective(hparams, budget)
        self._trials.append((hparams, budget, metric))
        self.report_best()
        return metric

    def report_best(self):
        trials = self._trials
        metrics = [r[2] for r in trials]
        i = int(np.argmin(metrics))
        n = len(trials)
        print(f"{n:>4} {i:>4} {metrics[i]:8.4f} {trials[i][0]}")

    def successive_halving(self, configs, budget, s):
        for i in range(s + 1):
            metrics = []
            for config in configs:
                m = self.run_trial(config, budget)
                metrics.append(m)
            budget = budget / self.eta
            configs = halving(configs, metrics, self.eta)

    def tune(self, objective, search_space):
        self._objective = objective
        self._search_space = search_space

        R, eta = self.R, self.eta
        s_max = int(math.log(self.R, eta))

        for s in range(s_max, -1, -1):
            n = round(int((s_max + 1) / (s + 1)) * (eta ** s))
            configs = self.sample_n(n)
            budget = R * (eta ** (-s))
            self.successive_halving(configs, budget, s)

    def reset(self):
        self._trials = []
        self._objective = None
        self._search_space = None
