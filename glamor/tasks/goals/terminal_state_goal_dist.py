from glamor.tasks.list_task_dist import ListTaskDist, RandomListTaskDist
from glamor.tasks.base import NoneTaskDist
from glamor.samplers.trajectory_sampler import TrajectorySampler
from collections import namedtuple

from tqdm import tqdm

Goal = namedtuple('Goal', ['obs', 'info'])

def generate_goals(env_cls, horizon, policy, n_goals, include_terminal, random_horizon):
	tasks = NoneTaskDist()
	sampler = TrajectorySampler(env_cls=env_cls,
                                policy=policy,
                                horizon=horizon,
                                tasks=tasks,
                                lazy_labels=True,
                                random_horizon=random_horizon)
	goals_collected = 0
	goals = []

	prog = tqdm(total=n_goals)
	while len(goals) < n_goals:
		traj = sampler.collect_trajectories(n_interactions=None, 
			                                n_trajs=1)[0]
		if include_terminal or len(traj.obs) == horizon:
			goals.append(Goal(obs=traj.obs[-1], info=traj.infos[-1]))
			prog.update(1)

	prog.close()
	return goals

class ListTerminalStateGoalDist(ListTaskDist):
	def __init__(self, env_cls, horizon, policy, n_goals, include_terminal=False, random_horizon=False):
		tasks = generate_goals(env_cls, 
			                   horizon, 
			                   policy, 
			                   n_goals, 
			                   include_terminal=include_terminal, 
			                   random_horizon=random_horizon)
		super().__init__(tasks)
		self._name = 'list_terminal_state'

class RandomTerminalStateGoalDist(RandomListTaskDist):
	def __init__(self, env_cls, horizon, policy, n_goals, include_terminal=False, random_horizon=False):
		tasks = generate_goals(env_cls, 
			                   horizon, 
			                   policy, 
			                   n_goals, 
			                   include_terminal=include_terminal, 
			                   random_horizon=random_horizon)
		super().__init__(tasks)
		self._name = 'random_terminal_state'