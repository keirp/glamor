from glamor.policies.base import BasePolicy, PolicyStep
from collections import namedtuple

import torch
import numpy as np

PolicyInfo = namedtuple("PolicyInfo", [])


class MixturePolicy(BasePolicy):
    """Takes a list of policies and a list of probabilities. Can be used to make
    epsilon-greedy for example by combining random and another policy."""

    def __init__(self, policies, probs):
        self.policies = policies
        self.probs = probs
        sub_policy_names = [policy.name for policy in policies]
        self._name = f'{"_".join(sub_policy_names)}_mixture'

    def reset(self):
        for policy in self.policies:
            policy.reset()

    def sample(self, obs, task, t):
        idx = np.random.choice(len(self.policies), p=self.probs)
        policy_res = [policy.sample(obs, task, t) for policy in self.policies]
        return policy_res[idx]
