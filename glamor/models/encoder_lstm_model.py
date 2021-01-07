import torch.nn as nn
import torch
from glamor.models.basic.mlp import MLP
import torch.nn.functional as F
	
class EncoderLSTMModel(nn.Module):
	"""A model that:
	- encodes observations using `obs_encoder`
	- encodes the task using `task_encoder` (usually the same as the obs_encoder)
	- uses an MLP to create a joint obs-task representation
	- sets the hidden state of the LSTM using this representation
	- predicts actions auto-regressively using the LSTM"""

	def __init__(self,
		         obs_shape,
		         n_actions,  
		         device,
		         obs_encoder,
		         task_encoder,
		         obs_rep_size,
		         task_rep_size,
		         state_size,
		         lstm_hidden_dim=64,
		         lstm_layers=1,
		         lstm_dropout_p=0,
		         ):
		super().__init__()
		self.device = device
		self.obs_shape = obs_shape
		self.n_actions = n_actions
		self.lstm_layers = lstm_layers
		self.lstm_hidden_dim = lstm_hidden_dim

		self.obs_encoder = obs_encoder
		self.task_encoder = task_encoder
		self.obs_rep_size = obs_rep_size
		self.task_rep_size = task_rep_size
		self.state_size = state_size

		print(f'Obs rep size: {self.obs_rep_size}, Task rep size: {self.task_rep_size}, State size: {self.state_size}')

		self.state_mix_nn = nn.Sequential(MLP(in_dim=self.obs_rep_size + self.task_rep_size,
										      out_dim=self.state_size,
										      hidden=[256],
										      use_layer_norm=True,
										      nonlinearity=nn.ReLU),
		                                  nn.ReLU())

		self.init_h = nn.Linear(self.state_size, lstm_hidden_dim)
		self.init_c = nn.Linear(self.state_size, lstm_hidden_dim)

		self.baseline_init_h = nn.Linear(self.obs_rep_size, lstm_hidden_dim)
		self.baseline_init_c = nn.Linear(self.obs_rep_size, lstm_hidden_dim)

		# Note: n_actions + 2 is for <HERE> and <PAD> tokens
		self.input_mix_nn = nn.Linear(self.state_size + n_actions, self.state_size)
		self.baseline_input_mix_nn = nn.Linear(self.obs_rep_size + n_actions, self.state_size)

		self.action_lstm = nn.LSTM(self.state_size, lstm_hidden_dim, batch_first=True, dropout=lstm_dropout_p, num_layers=lstm_layers)

		self.baseline_lstm = nn.LSTM(self.state_size, lstm_hidden_dim, batch_first=True, dropout=lstm_dropout_p, num_layers=lstm_layers)

		self.action_decoder = nn.Linear(lstm_hidden_dim, n_actions + 2)
		self.baseline_action_decoder = nn.Linear(lstm_hidden_dim, n_actions + 2)

		# self.action_decoder = MLP(in_dim=lstm_hidden_dim,
		# 					      out_dim=n_actions + 2,
		# 					      hidden=[256],
		# 					      use_layer_norm=True,
		# 					      nonlinearity=nn.ReLU)
		# self.baseline_action_decoder = MLP(in_dim=lstm_hidden_dim,
		# 					               out_dim=n_actions + 2,
		# 					               hidden=[256],
		# 					               use_layer_norm=True,
		# 					               nonlinearity=nn.ReLU)

	def init_hidden_state(self, state_mix):
		batch_size = state_mix.shape[0]
		
		hidden = (torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden_dim).to(device=self.device),
			      torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden_dim).to(device=self.device))
		h = self.init_h(state_mix)
		c = self.init_c(state_mix)

		hidden[0][0], hidden[1][0] = h, c
		return hidden

	def init_hidden_state_baseline(self, states_0):
		batch_size = states_0.shape[0]
		
		hidden = (torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden_dim).to(device=self.device),
			      torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden_dim).to(device=self.device))
		h = self.baseline_init_h(states_0)
		c = self.baseline_init_c(states_0)

		hidden[0][0], hidden[1][0] = h, c
		return hidden

	def get_state_mix(self, obs, tasks):
		obs_rep = F.relu(self.obs_encoder(obs).view(-1, self.obs_rep_size))
		task_rep = F.relu(self.task_encoder(tasks).view(-1, self.task_rep_size))

		# generate state diff vector
		states = torch.cat([obs_rep, task_rep], dim=-1)
		state_mix = self.state_mix_nn(states)

		return state_mix, obs_rep

	def pred_actions(self, obs, tasks, actions):
		batch_size = obs.shape[0]
		km1 = actions.shape[1]

		# truncate <HERE> and <PAD> tokens from input
		actions = actions[:, :, :-2]

		state_mix, obs_rep = self.get_state_mix(obs, tasks)

		# pad actions with <START> token
		zero_action = torch.zeros(batch_size, 1, actions.shape[-1]).to(device=self.device)
		actions = torch.cat([zero_action, actions[:, :-1]], dim=1)

		# Predict goal conditioned actions
		hidden = self.init_hidden_state(state_mix)

		# generate LSTM inputs
		state_mix = state_mix.view(-1, 1, self.state_size).expand(-1, km1, self.state_size)
		state_actions = torch.cat([state_mix, actions], dim=-1)

		input_mix = self.input_mix_nn(state_actions)

		# generate action predictions
		output, _ = self.action_lstm(input_mix, hidden)

		p_actions = self.action_decoder(output).log_softmax(dim=-1)

		# Predict baseline actions
		hidden = self.init_hidden_state_baseline(obs_rep)

		# generate LSTM inputs
		obs_rep = obs_rep.view(-1, 1, self.obs_rep_size).expand(-1, km1, self.obs_rep_size)
		state_actions = torch.cat([obs_rep, actions], dim=-1)

		input_mix = self.baseline_input_mix_nn(state_actions)

		# generate action predictions
		output, _ = self.baseline_lstm(input_mix, hidden)

		p_actions_baseline = self.baseline_action_decoder(output).log_softmax(dim=-1)

		return p_actions, p_actions_baseline

	def pred_score_one_step(self, obs_rep, state_mix, prev_action, hiddens=None):
		"""Get the score p(a|s, task)/p(a|s) for a single step and return
		the hidden state of the LSTM"""

		batch_size = obs_rep.shape[0]

		if hiddens is None:
			baseline_hidden = self.init_hidden_state_baseline(obs_rep)
			hidden = self.init_hidden_state(state_mix)
			hiddens = [baseline_hidden, hidden]

		prev_action = prev_action[:, :, :-2]

		obs_rep = obs_rep.view(-1, 1, self.obs_rep_size)
		obs_actions = torch.cat([obs_rep, prev_action], dim=-1)

		state_mix = state_mix.view(-1, 1, self.state_size)
		state_actions_mix = torch.cat([state_mix, prev_action], dim=-1)

		input_mix = self.baseline_input_mix_nn(obs_actions)
		output, hiddens[0] = self.baseline_lstm(input_mix, hiddens[0])
		p_actions_baseline = self.baseline_action_decoder(output).view(batch_size, -1)[:, :-1]

		input_mix = self.input_mix_nn(state_actions_mix)
		output, hiddens[1] = self.action_lstm(input_mix, hiddens[1])
		p_actions = self.action_decoder(output).view(batch_size, -1)[:, :-1]

		p_actions = p_actions.log_softmax(dim=-1)
		p_actions_baseline = p_actions_baseline.log_softmax(dim=-1)
		score = p_actions - p_actions_baseline
		info = {'p_actions': p_actions, 'p_actions_baseline': p_actions_baseline}

		return score, hiddens, info

	def policy(self, obs, tasks):
		batch_size = obs.shape[0]
		state_mix, obs_rep = self.get_state_mix(obs, tasks)

		zero_action = torch.zeros(batch_size, 1, self.n_actions).to(device=self.device)
		hidden = self.init_hidden_state(state_mix)

		state_mix = state_mix.view(-1, 1, self.state_size)
		state_actions = torch.cat([state_mix, zero_action], dim=-1)

		input_mix = self.input_mix_nn(state_actions)

		output, _ = self.action_lstm(input_mix, hidden)

		p_actions = self.action_decoder(output).view(batch_size, -1)[:, :-2].softmax(dim=-1)

		return p_actions