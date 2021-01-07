from glamor.planner.base import BasePlanner, PlanResult

import torch
import numpy as np
from collections import defaultdict

class SamplingPlanner(BasePlanner):
	"""Approximately finds argmax_a p(a|s, task)/p(a|s) by sampling from the
	softmax distribution of the decomposed probabilities for each action
	many times and taking the max."""

	def __init__(self, model, gamma, n_actions, device, num_trials, clip_p_actions=None, use_prior=True, early_terminate=True):
		super().__init__(model, gamma, n_actions, device)
		self.num_trials = num_trials
		self.clip_p_actions = clip_p_actions
		self.use_prior = use_prior
		self.early_terminate = early_terminate

	def sample_action(self, p_actions):
		action = torch.multinomial(p_actions, num_samples=1)
		return action

	def sample_trajectory(self, state_mix, obs_rep, horizon, log=False):
		batch_size = state_mix.shape[0]
		prev_action = None
		t = 0
		hiddens = None
		running_score = torch.zeros(batch_size).to(device=self.device)
		log_scores = torch.zeros(batch_size).to(device=self.device)
		infos = [defaultdict(list) for _ in range(batch_size)]
		dones = [False] * batch_size
		traj = []

		invalid = [False] * batch_size
		ends = torch.zeros(batch_size).int().to(device=self.device)

		while t < horizon:
			prev_action_one_hot = torch.zeros(batch_size, self.n_actions + 2).to(device=self.device)
			if prev_action is not None:
				prev_action_one_hot = prev_action_one_hot.scatter_(1, prev_action, 1)
			prev_action_one_hot = prev_action_one_hot.view(batch_size, 1, -1)

			score, hiddens, info = self.model.pred_score_one_step(obs_rep, state_mix, prev_action_one_hot, hiddens)

			if not self.use_prior:
				score = info['p_actions']

			if self.clip_p_actions is not None:
				p_actions = info['p_actions']
				score[p_actions < self.clip_p_actions] = -20.

			sampling_dist = score.softmax(dim=-1)

			info['score'] = score
			info['sampling_dist'] = sampling_dist

			if t == 0 or not self.early_terminate:
				sampling_dist[:, -1] = 0
				sampling_dist /= sampling_dist.sum(dim=-1, keepdim=True)

			if t == horizon - 1:
				prev_action = torch.ones_like(prev_action).to(device=self.device) * self.end_token
			else:
				prev_action = self.sample_action(sampling_dist)

			sampled_score = score.gather(1, prev_action).view(batch_size)
			running_score += sampled_score

			for key in info:
				if info[key].shape[1] > 1:
					selected = info[key].gather(1, prev_action).view(batch_size)
				else:
					selected = info[key].view(batch_size)
				for i in range(batch_size):
					if not dones[i]:
						infos[i][key].append(selected[i].item())

			traj.append(prev_action)
			t += 1
			for i in range(batch_size):
				if not dones[i]:
					if prev_action[i].item() == self.end_token:
						ends[i] = t
						log_scores[i] = running_score[i] + t * np.log(self.gamma)
						traj_info = [t[i] for t in traj][:t]
						infos[i]['traj'] = traj_info
						infos[i]['t'] = t
						infos[i]['done'] = True
						dones[i] = True
					if t == horizon and not dones[i]:
						infos[i]['done'] = False
						infos[i]['t'] = t
						ends[i] = t
						log_scores[i] = np.NINF

		return traj, log_scores, ends, infos

	def plan(self, obs, task, horizon):
		self.model.eval()
		with torch.no_grad():
			state_mix, obs_rep = self.model.get_state_mix(obs, task)

			state_mix = state_mix.expand(self.num_trials, -1)
			obs_rep = obs_rep.expand(self.num_trials, -1)

			traj, log_scores, ends, infos = self.sample_trajectory(state_mix, obs_rep, horizon)
			best_score, idx = torch.max(log_scores, dim=0)
			action = traj[0][idx]
			traj_len = ends[idx]
			traj = [t[idx] for t in traj][:traj_len]

		return PlanResult(traj=traj, info={'score': best_score,
										   'len': traj_len,
										   'infos': infos})
