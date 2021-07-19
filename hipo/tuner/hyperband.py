from typing import List, Dict

import math
from collections import OrderedDict

import numpy as np

from hipo.sample import sample
from hipo.trial import Trial
from hipo.utils import datetime_now


def halving(configs, losses, eta):
    n = len(configs)
    k = int(n / eta)
    indices = np.argpartition(losses, k)[:k]
    return [configs[i] for i in indices]


class HyperbandTuner:

    def __init__(self, R, eta):
        self.R = R
        self.eta = eta

        self._trials: Dict[int, Trial] = OrderedDict()
        self._objective = None
        self._search_space = None

        self._best_value = np.inf
        self._best_trial_id = None

    def reset(self):
        self._trials: Dict[int, Trial] = OrderedDict()
        self._objective = None
        self._search_space = None

        self._best_value = np.inf
        self._best_trial_id = None


    def after_trial(self):
        self.update_best()

    def sample(self):
        return sample(self._search_space)

    def sample_n(self, n):
        return [self.sample() for i in range(n)]

    def run_trial(self, hparams, budget):
        start = datetime_now()
        value = self._objective(hparams, budget)
        trial_id = len(self._trials)
        trial = Trial(
            id=trial_id, params=hparams, budget=budget, value=value,
            start=start, end=datetime_now())
        self._trials[trial_id] = trial
        self.after_trial()
        return trial_id

    def update_best(self):
        last_trial_id = next(reversed(self._trials))
        last_trial = self._trials[last_trial_id]
        if last_trial.value >= self._best_value:
            return
        self._best_value = last_trial.value
        self._best_trial_id = last_trial_id
        n = len(self._trials)
        print(f"{last_trial_id:>4} {last_trial.value:8.4f} {last_trial.params}")

    def successive_halving(self, configs, budget, s):
        for i in range(s + 1):
            metrics = []
            for config in configs:
                trial_id = self.run_trial(config, budget)
                value = self._trials[trial_id].value
                metrics.append(value)
            budget = budget * self.eta
            configs = halving(configs, metrics, self.eta)

    def tune(self, objective, search_space):
        self._objective = objective
        self._search_space = search_space

        R, eta = self.R, self.eta
        n_brackets = int(math.log(self.R, eta))

        for s in range(n_brackets, -1, -1):
            n = round(int((n_brackets + 1) / (s + 1)) * (eta ** s))
            configs = self.sample_n(n)
            budget = R * (eta ** (-s))
            self.successive_halving(configs, budget, s)