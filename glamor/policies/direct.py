from glamor.policies.base import BasePolicy, PolicyStep
from collections import namedtuple

import torch

PolicyInfo = namedtuple("PolicyInfo", [])


class DirectPolicy(BasePolicy):
    """Takes a policy model (obs, task -> action dist) and samples
    from it to pick an action."""

    def __init__(self, model, horizon, device):
        self.model = model
        self._name = 'direct'
        self.device = device
        self.horizon = horizon

    def _sample_action(self, p_actions):
        action = torch.multinomial(p_actions, num_samples=1)
        return action

    def sample(self, obs, task, t):
        obs = torch.tensor(obs).unsqueeze(0).float().to(device=self.device)
        task = torch.tensor(task.obs).unsqueeze(
            0).float().to(device=self.device)

        ks = torch.tensor(
            self.horizon - t).unsqueeze(0).unsqueeze(0).float().to(device=self.device)

        p_actions = self.model(obs, task, ks).view(1, -1)

        action = self._sample_action(p_actions)
        return PolicyStep(action=action.item(), info=PolicyInfo())
