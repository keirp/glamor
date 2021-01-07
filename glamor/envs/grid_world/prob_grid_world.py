import gym
from gym import spaces
import numpy as np
from random import random

class ProbGridWorld(gym.Env):

	def __init__(self, w, h, probs=[1, 0.9]):
		self.h = h
		self.w = w
		self.x = w // 2
		self.y = h // 2

		self.n_groups = len(probs)
		self.probs = probs

		self.action_space = spaces.Discrete(4 * self.n_groups)
		self.observation_space = spaces.Box(low=0,
											high=255, 
											shape=(1, self.w, self.h), 
											dtype=np.int)
	
	def step(self, action):
		direction = action % 4
		prob = self.probs[action // 4]
		if random() < prob:
			if direction == 0 and self.y < self.h - 1:
				self.y += 1
			elif direction == 1 and self.x < self.w - 1:
				self.x += 1
			elif direction == 2 and self.y > 0:
				self.y -= 1
			elif direction == 3 and self.x > 0:
				self.x -= 1
		else:
			# stand still
			pass

		return self._get_grid(), 0, False, {}

	def _get_grid(self):
		grid = np.zeros((1, self.w, self.h), dtype=np.int)
		grid[0, self.x, self.y] = 255
		return grid

	def reset(self):
		self.x = self.w // 2
		self.y = self.h // 2

		return self._get_grid()

	def close(self):
		pass

	def render(self):
		pass
