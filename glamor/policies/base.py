from collections import namedtuple

PolicyStep = namedtuple("PolicyStep", ["action", "info"])


class BasePolicy:

    def reset(self):
        pass

    def sample(self, obs, task, t):
        raise NotImplementedError

    def update(self, total_interactions):
        """Use this to update epsilon"""
        pass

    @property
    def name(self):
        return self._name
