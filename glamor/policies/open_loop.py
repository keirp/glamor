from glamor.policies.base import BasePolicy, PolicyStep
from collections import namedtuple

import torch

PolicyInfo = namedtuple("PolicyInfo", ["time_to_go"])


class OpenLoopPolicy(BasePolicy):
    """Uses a planner at the first timestep to compute a plan, 
    then takes the actions in that plan one by one without
    recomputing based on new observations.

    After the plan runs out, take random actions"""

    def __init__(self, planner, action_space, terminate, horizon, device):
        self.planner = planner
        self.plan = []
        self.t = 0
        self.action_space = action_space
        self._name = 'open_loop'
        self.device = device
        self.terminate = terminate
        self.horizon = horizon

    def reset(self):
        self.plan = []
        self.t = 0

    def sample(self, obs, task, t):
        if len(self.plan) == 0:
            obs = torch.tensor(obs).unsqueeze(0).float().to(device=self.device)
            task = torch.tensor(task.obs).unsqueeze(
                0).float().to(device=self.device)

            res = self.planner.plan(obs, task, self.horizon)

            self.plan = res.traj

        if self.t > len(self.plan) - 2:
            if self.terminate:
                return PolicyStep(action=self.planner.end_token, info=PolicyInfo(time_to_go=0))
            return PolicyStep(action=self.action_space.sample().item(), info=PolicyInfo(time_to_go=0))

        action = self.plan[self.t]
        self.t += 1

        return PolicyStep(action=action.item(), info=PolicyInfo(time_to_go=(len(self.plan) - self.t) / self.horizon))
