import numpy as np
import torch

from collections import namedtuple

Sample = namedtuple("Sample", ["obs", "obs_k", "actions", "ks"])

class ReplayBuffer:

	def __init__(self,
		         buffer_size,
		         env_cls,
		         k_dist,
		         end_token,
		         frames_goal=True,
		         action_seq=True):

		self.k_dist = k_dist
		self.end_token = end_token
		self.env_cls = env_cls
		env = env_cls()

		self.t = 0
		self.current_buffer_size = 0
		self.max_buffer_size = buffer_size

		print(f'Replay buffer with obs_buffer shape {(buffer_size, *env.observation_space.shape)}')
		print(f'Datatypes: obs: {env.observation_space.dtype}, actions: {env.action_space.dtype}')

		self.obs_buffer = np.zeros((buffer_size, *env.observation_space.shape),
		                           dtype=env.observation_space.dtype)
		self.action_buffer = np.zeros((buffer_size, *env.action_space.shape),
			                          dtype=env.action_space.dtype)
		self.traj_end_buffer = np.zeros((buffer_size),
			                            dtype=np.int32)
		self.t_buffer = np.zeros((buffer_size), dtype=np.int32)
		self.frames_goal = frames_goal
		self.action_seq = action_seq

	def __len__(self):
		return self.current_buffer_size

	def append_trajs(self, trajs):
		"""Append a list of traj objects"""
		for traj in trajs:
			T = len(traj.obs)
			stacked_obs = np.stack(traj.obs, axis=0)
			actions = np.array(traj.actions)

			idxs = np.arange(self.t, self.t + T) % self.max_buffer_size
			self.obs_buffer[idxs] = stacked_obs
			self.action_buffer[idxs] = actions
			self.traj_end_buffer[idxs] = idxs[-1]
			self.t_buffer[idxs] = np.arange(T)
			self.t = (self.t + T) % self.max_buffer_size

			if self.current_buffer_size < self.max_buffer_size:
				self.current_buffer_size = min(self.current_buffer_size + T, self.max_buffer_size)

	def sample(self, n, idxs=None):
		"""Sample n (s, sg, a) pairs and return numpy objects"""
		if idxs is None:
			idxs = np.random.randint(low=0, high=self.current_buffer_size, size=(n,))

		ks = np.array([self.k_dist.sample() for _ in range(n)])
		len_remainings = (self.traj_end_buffer[idxs] - idxs) % self.max_buffer_size

		ks = np.minimum(ks, len_remainings)

		future_idxs = (idxs + ks) % self.max_buffer_size
		obs = self.obs_buffer[idxs]

		if self.frames_goal:
			obs_k = self.obs_buffer[future_idxs]
		else:
			obs_k = self.obs_buffer[future_idxs][:, -1][:, np.newaxis]

		

		# Can't figure out how to do this vectorized...
		if self.action_seq:
			actions_one_hot = np.zeros((n, self.k_dist.max + 1, self.end_token + 2), dtype=self.action_buffer.dtype)

			for i in range(n):
				a_idx = np.arange(idxs[i], idxs[i] + ks[i]) % self.max_buffer_size
				actions = np.ones(self.k_dist.max + 1, dtype=self.action_buffer.dtype) * (self.end_token + 1)
				actions[:ks[i]] = self.action_buffer[a_idx]
				actions[ks[i]] = self.end_token

				actions_one_hot[i, np.arange(actions.size), actions] = 1
		else:
			actions_one_hot = np.zeros((n, self.end_token), dtype=self.action_buffer.dtype)
			actions = self.action_buffer[idxs]
			actions_one_hot[np.arange(actions.size), actions] = 1

		return Sample(obs=torch.tensor(obs), 
			          obs_k=torch.tensor(obs_k), 
			          actions=torch.tensor(actions_one_hot),
			          ks=torch.tensor(ks))