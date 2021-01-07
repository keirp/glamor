from collections import namedtuple

PlanResult = namedtuple("PlanResult", ["traj", "info"])

class BasePlanner:

	def __init__(self, model, gamma, n_actions, device):
		self.gamma = gamma
		self.n_actions = n_actions
		self.end_token = n_actions
		self.device = device
		self.model = model

	def plan(self, obs, task):
		raise NotImplementedError