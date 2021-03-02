from glamor.policies.base import BasePolicy, PolicyStep
from glamor.policies.mixture import MixturePolicy
from glamor.policies.random import RandomPolicy
from collections import namedtuple


class EpsilonGreedyPolicy(MixturePolicy):
    """Implements epsilon-greedy policy as a
    mixture policy with support for a decreasing
    epsilon schedule"""

    def __init__(self, policy, final_eps, eps_steps, action_space):
        policies = [policy, RandomPolicy(action_space=action_space)]
        super().__init__(policies=policies, probs=[0, 1])
        self.final_eps = final_eps
        self.eps_steps = eps_steps
        self._name = 'eps_greedy'

    def update(self, total_interactions):
        if total_interactions < self.eps_steps:
            prog = (total_interactions / self.eps_steps)
            eps = prog * self.final_eps +\
                (1 - prog)
        else:
            eps = self.final_eps
        self.probs = [1 - eps, eps]
