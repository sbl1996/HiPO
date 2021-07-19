from typing import Dict

from collections import OrderedDict

import numpy as np

from hipo.trial import Trial
from hipo.utils import datetime_now


def halving(configs, losses, eta):
    n = len(configs)
    k = int(n / eta)
    indices = np.argpartition(losses, k)[:k]
    return [configs[i] for i in indices]


class BaseTuner:

    def __init__(self):

        self._trials: Dict[int, Trial] = OrderedDict()
        self._objective = None
        self._search_space = None

        self._best_value = np.inf
        self._best_trial_id = None
        self._max_trials = None

    def reset(self):
        self._trials: Dict[int, Trial] = OrderedDict()
        self._objective = None
        self._search_space = None

        self._best_value = np.inf
        self._best_trial_id = None
        self._max_trials = None

    def after_trial(self):
        self.update_best()

    def sample(self):
        raise NotImplementedError

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
        print(f"{last_trial_id:>4} {last_trial.value:8.4f} {last_trial.budget:.2f} {last_trial.params}")

    def tune(self, objective, search_space, max_trials=None):
        self._objective = objective
        self._search_space = search_space