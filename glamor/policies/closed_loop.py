from glamor.policies.base import BasePolicy, PolicyStep
from collections import namedtuple

import torch

PolicyInfo = namedtuple("PolicyInfo", ["time_to_go"])

class ClosedLoopPolicy(BasePolicy):
	"""Uses a planner at each step to compute a plan, then takes
	the first action in that plan."""

	def __init__(self, planner, action_space, horizon, device, terminate=True):
		self.planner = planner
		self._name = 'closed_loop'
		self.horizon = horizon
		self.device = device
		self.action_space = action_space
		self.end = False
		self.terminate = terminate

	def reset(self):
		self.end = False

	def sample(self, obs, task, t):
		if self.end and self.terminate:
			return PolicyStep(action=self.planner.end_token, info=PolicyInfo(time_to_go=0))
		obs = torch.tensor(obs).unsqueeze(0).float().to(device=self.device)
		task = torch.tensor(task.obs).unsqueeze(0).float().to(device=self.device)

		res = self.planner.plan(obs, task, self.horizon - t)
		action = res.traj[0]

		if not self.terminate:
			if action == self.planner.end_token:
				# if the traj is supposed to end, just sample
				# random action
				action = self.action_space.sample().item()

		if res.traj[1] == self.planner.end_token:
			self.end = True

		return PolicyStep(action=action.item(), info=PolicyInfo(time_to_go=len(res.traj) / self.horizon))
