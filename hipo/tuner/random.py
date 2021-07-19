from hipo.tuner.base import BaseTuner
from hipo.sample import sample


class RandomTuner(BaseTuner):

    def __init__(self, budget):
        super().__init__()
        self.budget = budget

    def sample(self):
        return sample(self._search_space)

    def tune(self, objective, search_space, max_trials=100):
        self._objective = objective
        self._search_space = search_space

        while len(self._trials) < max_trials:
            params = self.sample()
            self.run_trial(params, self.budget)