import gym
from gym import spaces
import numpy as np
from random import random

class ProbEnv(gym.Env):

	def __init__(self):
		self.state = 0

		self.action_space = spaces.Discrete(2)
		self.observation_space = spaces.Box(low=0,
											high=255, 
											shape=[3], 
											dtype=np.int)
	
	def step(self, action):
		terminal = False
		if self.state == 0:
			if action == 0:
				if random() < 0.8:
					self.state = 1
				else:
					self.state = 2
			if action == 1:
				if random() < 0.5:
					self.state = 1
				else:
					self.state = 2
		else:
			terminal = True

		return self._get_obs(), 0, False, {}

	def _get_obs(self):
		one_hot = np.zeros(3, dtype=np.int)
		one_hot[self.state] = 255
		return one_hot

	def reset(self):
		self.state = 0

		return self._get_obs()

	def close(self):
		pass

	def render(self):
		pass
