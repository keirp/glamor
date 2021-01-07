import numpy as np
import os
from rlpyt.envs.base import Env, EnvStep
from rlpyt.spaces.int_box import IntBox
from collections import namedtuple
from random import random


from glamor.tasks.list_task_dist import ListTaskDist, RandomListTaskDist

EnvInfo = namedtuple("EnvInfo", ["traj_done", "labels", "goal_labels"])
Goal = namedtuple('Goal', ['obs', 'info'])
GridWorldLabels = namedtuple('GridWorldLabels', ['x', 'y'])

class CustomGridWorld(Env):
	""" Grid World where some cell locations may be a wall. 
	"""

	def __init__(self, start_x, start_y, grid_file):
		self.x, self.y = start_x, start_y
		self.start_x, self.start_y = start_x, start_y

		curr_abs_path = os.path.dirname(os.path.abspath(__file__))
		rel_path = os.path.join(curr_abs_path, "worlds", grid_file)

		if os.path.exists(rel_path):
			grid_file = rel_path
			self.grid = np.loadtxt(grid_file, delimiter=',')
			# Overwrite grid size if necessary
			self.h = self.grid.shape[0]
			self.w = self.grid.shape[1]

		self._action_space = IntBox(low=0, high=4)
		self._observation_space = IntBox(low=0, high=255, shape=(2, self.w, self.h), dtype="uint8")


	def step(self, action):
		if action == 0 and self.y < self.h - 1:
			if self.grid[self.x, self.y + 1] != 2:
				self.y += 1
		elif action == 1 and self.x < self.w - 1:
			if self.grid[self.x + 1, self.y] != 2:
				self.x += 1
		elif action == 2 and self.y > 0:
			if self.grid[self.x, self.y - 1] != 2:
				self.y -= 1
		elif action == 3 and self.x > 0:
			if self.grid[self.x - 1, self.y] != 2:
				self.x -= 1
		else:
			# stand still
			pass

		info = EnvInfo(traj_done=False, labels=None, goal_labels=None)
		return EnvStep(self._get_grid(), 0, False, info)
		# return (self.x, self.y), 0, False, {}

	def labels(self):
		labels = {'x': self.x, 'y': self.y}
		return GridWorldLabels(**labels)

	def _get_grid(self):
		grid = np.zeros((2, self.h, self.w), dtype=np.int)
		grid[0, self.x, self.y] = 255
		grid[1, self.grid == 2] = 255
		return grid

	def reset(self):
		self.x, self.y = self.start_x, self.start_y

		return self._get_grid()

	def close(self):
		pass

	def render(self):
		pass

class UniformGridWorldGoalDist(RandomListTaskDist):

	def __init__(self, grid):

		# init namedtuple
		labels = {'x': 0, 'y': 0}

		tasks = []

		for x in range(grid.shape[0]):
			for y in range(grid.shape[1]):
				if grid[x, y] == 0:
					task_grid = np.zeros((2, grid.shape[0], grid.shape[1]), dtype=np.int)
					task_grid[1, grid == 2] = 255
					task_grid[0, x, y] = 255
					info = EnvInfo(traj_done=False, labels=GridWorldLabels(x=x, y=y), goal_labels=None)
					tasks.append(Goal(obs=task_grid, info=info))

		super().__init__(tasks)
		self._name = 'grid_world_goals'
