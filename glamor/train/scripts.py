from glamor.envs.atari.annotated_atari_env import AnnotatedAtariEnv

try:
	from glamor.envs.dm_control.annotated_dm_control import AnnotatedDMControl
except:
	pass

from glamor.models.encoder_lstm_model import EncoderLSTMModel
from glamor.models.basic.residule import Impala
from glamor.models.atari.atari_encoder import AtariEncoder, AtariSingleFrameEncoder

from glamor.planner.sampling import SamplingPlanner

from glamor.policies.random import RandomPolicy
from glamor.policies.open_loop import OpenLoopPolicy
from glamor.policies.closed_loop import ClosedLoopPolicy
from glamor.policies.direct import DirectPolicy
from glamor.policies.mixture import MixturePolicy
from glamor.policies.eps_greedy import EpsilonGreedyPolicy

from glamor.tasks.goals.terminal_state_goal_dist import ListTerminalStateGoalDist, RandomTerminalStateGoalDist
from glamor.tasks.goals.diverse_label_goal_dist import ListDiverseLabelGoalDist, RandomDiverseLabelGoalDist
from glamor.tasks.base import NoneTaskDist

from glamor.eval.policy_video_eval import PolicyVideoEval
from glamor.eval.label_compare_eval import LabelCompareEval

from glamor.datasets.k_dist import UniformK

from glamor.algos.batch_train_glamor import BatchTrainGLAMOR

from glamor.train.config.labels import included_labels

from glamor.utils.checkpoint import restore_wandb_id, commit_wandb_id

import torch
import wandb
import numpy as np

from glamor.utils.logging import disable_wandb, wandb_is_enabled, set_run_path

def train_glamor(env_cls,
			     model,
			     policy='open_loop',
			     planner='sampling',
			     final_eps=0.1,
			     eps_steps=int(1e6),
			     n_interactions=int(5e6),
			     k=50,
			     lr=1e-3,
			     policy_trials=200,
			     clip_p_actions=None,
			     use_prior=False,
			     train_tasks=None,
			     test_policies=['open_loop', 'closed_loop', 'direct'],
			     test_tasks=['random', 'diverse'],
			     eval_types=['labels', 'video'],
			     n_label_tasks=5,
			     n_video_tasks=5,
			     include_labels=None,
			     n_eval_tasks=5,
			     device='cuda',
			     gamma=0.99,
			     n_tasks=1000,
			     replay_ratio=4,
			     log_period=int(1e5),
			     snapshot_period=int(5e5),
			     buffer_size=int(1e6),
			     min_step_learn=int(1e5),
			     frame_buffer=True,
			     checkpoint_path=None,
			     ):
	assert policy in ['uniform', 'open_loop']

	env = env_cls()
	obs = env.reset()
	obs_shape = obs.shape
	n_actions = env.action_space.n
	print('Initialized environment.')

	# planner
	if clip_p_actions is None:
		clip = None
	else:
		clip = clip_p_actions

	if planner == 'sampling':
		planner = SamplingPlanner(model=model, 
								  gamma=gamma,
								  n_actions=n_actions, 
								  device=device,
								  num_trials=policy_trials,
								  clip_p_actions=clip,
								  use_prior=use_prior)
	else:
		raise NotImplementedError

	# training policy
	if policy == 'uniform':
		training_policy = RandomPolicy(action_space=env.action_space)
	elif policy == 'open_loop':
		# open_loop with epsilon-greedy exploration
		open_loop_policy = OpenLoopPolicy(planner=planner,
										  action_space=env.action_space,
										  terminate=False, 
										  horizon=k + 1,
										  device=device)
		training_policy = EpsilonGreedyPolicy(policy=open_loop_policy,
											  final_eps=final_eps,
											  eps_steps=eps_steps,
											  action_space=env.action_space)

	# tasks

	if include_labels is None:
		include_fn = None
	else:
		include_fn = lambda key: key in include_labels

	if train_tasks is None:
		train_tasks = RandomDiverseLabelGoalDist(env_cls=env_cls, 
												 horizon=k, 
												 policy=RandomPolicy(action_space=env.action_space), 
												 n_goals=n_tasks, 
												 n_samples=n_tasks * 2,
												 include_fn=include_fn)

	eval_tasks = []
	for task_ in test_tasks:
		if task_ == 'random':
			eval_tasks.append(ListTerminalStateGoalDist(env_cls=env_cls, 
														horizon=k, 
														policy=RandomPolicy(action_space=env.action_space), 
														n_goals=n_eval_tasks, 
														include_terminal=False))
		elif task_ == 'diverse':
			eval_tasks.append(ListDiverseLabelGoalDist(env_cls=env_cls, 
													   horizon=k, 
													   policy=RandomPolicy(action_space=env.action_space), 
													   n_goals=n_eval_tasks, 
													   n_samples=200,
													   include_fn=include_fn))
		elif task_ == 'train':
			eval_tasks.append(train_tasks)
	
	print('Generated task distributions.')

	# eval planners
	eval_policies = []
	for policy_ in test_policies:
		if policy_ == 'open_loop':
			eval_policies.append(OpenLoopPolicy(planner=planner,
												action_space=env.action_space,
												terminate=True,
												horizon=k + 1,
												device=device))
		elif policy_ == 'closed_loop':
			eval_policies.append(ClosedLoopPolicy(planner=planner,
												  horizon=k + 1,
												  action_space=env.action_space,
												  terminate=True,
												  device=device))

	# evaluators

	eval_horizon = k

	evals = []

	for eval_policy in eval_policies:
		for eval_task in eval_tasks:
			time_to_go = eval_policy.name != 'direct'
			if 'video' in eval_types:
				evals.append(PolicyVideoEval(env_cls=env_cls, 
											 horizon=eval_horizon, 
											 policy=eval_policy, 
											 tasks=eval_task,
											 time_to_go=time_to_go,
											 n_trajs=n_video_tasks))
			if 'labels' in eval_types:
				evals.append(LabelCompareEval(env_cls=env_cls, 
											  horizon=eval_horizon, 
											  policy=eval_policy, 
											  tasks=eval_task, 
											  n_trajs=n_label_tasks,
											  include_labels=include_labels))
	# k distribution
	k_dist = UniformK(max_k=k)

	# algo
	algo = BatchTrainGLAMOR(env_cls=env_cls,
						    horizon=k,
						    k_dist=k_dist,
						    n_interactions=n_interactions,
						    replay_ratio=replay_ratio,
						    policy=training_policy,
						    task_dist=train_tasks,
						    model=model,
						    evals=evals,
						    batch_size=200,
						    log_period=log_period,
						    snapshot_period=snapshot_period,
						    buffer_size=buffer_size,
						    min_step_learn=min_step_learn,
						    lr=lr,
						    device=device,
						    frame_buffer=frame_buffer,
						    checkpoint_path=checkpoint_path)

	print('Started training.')
	algo.train()


def train_glamor_atari(game='pong',
				       policy='open_loop',
				       planner='sampling',
				       final_eps=0.1,
				       eps_steps=int(3e5),
				       n_interactions=int(1e6),
				       k=50,
				       state_size=512,
				       lstm_hidden_dim=64,
				       lstm_layers=1,
				       lstm_dropout_p=0,
				       encoder_dropout_p=0, # ADD TO CONTROL TOO
				       frames_goal=False,
				       lr=5e-4,
				       policy_trials=200,
				       clip_p_actions=None,
				       use_prior=False,
				       test_policies=['closed_loop'],
				       test_tasks=['train'],
				       eval_types=['labels', 'video'],
				       n_label_tasks=5,
				       n_video_tasks=5,
				       n_eval_tasks=20,
				       repeat_action_probability=0.25,
				       device='cuda',
				       gamma=0.99,
				       n_tasks=1000,
				       replay_ratio=4,
				       log_period=int(5e4),
				       snapshot_period=int(5e5),
				       buffer_size=int(3e5),
				       min_step_learn=int(5e4),
				       seed=0,
				       use_wandb=True,
				       run_path=None,
				       checkpoint_path=None):

	torch.backends.cudnn.benchmark = True

	torch.manual_seed(seed)

	if not use_wandb:
		disable_wandb(run_path)

	if wandb_is_enabled():
		if checkpoint_path is not None:
			wandb_id = restore_wandb_id(checkpoint_path)
			if wandb_id is None:
				wandb_id = wandb.util.generate_id()
			commit_wandb_id(checkpoint_path, wandb_id)
		else:
			wandb_id = wandb.util.generate_id()

		wandb.init(id=wandb_id,
			       resume='allow',
				   project='controllable',
		           entity='controllable', 
		           config=locals(), 
		           job_type='train_id_atari')

		set_run_path(wandb.run.dir)

	# env fixes
	min_start_noops = 0
	if game == 'ms_pacman':
		min_start_noops = 50 * 4
	elif game == 'venture':
		min_start_noops = 20 * 4
	elif game == 'space_invaders':
		min_start_noops = 20 * 4

	include_labels = included_labels[game]

	# env
	env_cls = lambda: AnnotatedAtariEnv(game=game, 
										repeat_action_probability=repeat_action_probability,
										min_start_noops=min_start_noops)
	env = env_cls()
	obs = env.reset()
	obs_shape = obs.shape
	n_actions = env.action_space.n
	print('Initialized environment.')

	# model
	if frames_goal:
		shared_encoder = Impala(obs_shape=obs_shape,
								state_size=state_size,
								dropout_p=encoder_dropout_p,
								group_norm=True)
		shared_encoder = AtariEncoder(encoder=shared_encoder)
		obs_encoder = shared_encoder
		task_encoder = shared_encoder
	else:
		obs_encoder = Impala(obs_shape=obs_shape,
							 state_size=state_size,
							 dropout_p=encoder_dropout_p,
							 group_norm=True)
		obs_encoder = AtariEncoder(encoder=obs_encoder)

		single_obs_shape = list(obs_shape)
		single_obs_shape[0] = 1
		task_encoder = Impala(obs_shape=single_obs_shape,
							  state_size=state_size,
							  dropout_p=encoder_dropout_p,
							  group_norm=True)
		task_encoder = AtariSingleFrameEncoder(encoder=task_encoder,
											   obs_shape=obs_shape)

	model = EncoderLSTMModel(obs_shape=obs_shape,
							 n_actions=n_actions,  
							 device=device,
							 obs_encoder=obs_encoder,
							 task_encoder=task_encoder,
							 obs_rep_size=state_size,
							 task_rep_size=state_size,
							 state_size=state_size,
							 lstm_hidden_dim=lstm_hidden_dim,
							 lstm_layers=lstm_layers,
							 lstm_dropout_p=lstm_dropout_p).to(device=device)
	print('Initialized models.')

	train_glamor(env_cls=env_cls,
			     model=model,
			     policy=policy,
			     planner=planner,
			     final_eps=final_eps,
			     eps_steps=int(eps_steps),
			     n_interactions=int(n_interactions),
			     k=k,
			     lr=lr,
			     policy_trials=policy_trials,
			     clip_p_actions=clip_p_actions,
			     use_prior=use_prior,
			     test_policies=test_policies,
			     test_tasks=test_tasks,
			     eval_types=eval_types,
			     include_labels=include_labels,
			     n_label_tasks=n_label_tasks,
			     n_video_tasks=n_video_tasks,
			     n_eval_tasks=n_eval_tasks,
			     device=device,
			     gamma=gamma,
			     n_tasks=n_tasks,
			     replay_ratio=replay_ratio,
			     log_period=int(log_period),
			     snapshot_period=int(snapshot_period),
			     buffer_size=int(buffer_size),
			     min_step_learn=int(min_step_learn),
			     checkpoint_path=checkpoint_path)

def train_glamor_control(env_name='point_mass/easy',
					     policy='open_loop',
					     planner='sampling',
					     final_eps=0.1,
					     eps_steps=int(3e5),
					     n_interactions=int(1e6),
					     k=100,
					     state_size=512,
					     lstm_hidden_dim=64,
					     lstm_layers=1,
					     lstm_dropout_p=0,
					     frames_goal=False,
					     lr=5e-4,
					     policy_trials=200,
					     clip_p_actions=None,
					     test_policies=['closed_loop'],
					     test_tasks=['train'],
					     eval_types=['labels', 'video'],
					     n_label_tasks=5,
					     n_video_tasks=5,
					     n_eval_tasks=20,
					     device='cuda',
					     gamma=0.99,
					     n_tasks=1000,
					     replay_ratio=4,
					     log_period=int(5e4),
					     snapshot_period=int(5e5),
					     buffer_size=int(3e5),
					     min_step_learn=int(5e4),
					     seed=0,
					     use_wandb=True,
				         run_path=None,
					     checkpoint_path=None):
	
	torch.backends.cudnn.benchmark = True

	torch.manual_seed(seed)

	if not use_wandb:
		disable_wandb(run_path)

	if wandb_is_enabled():

		if checkpoint_path is not None:
			wandb_id = restore_wandb_id(checkpoint_path)
			if wandb_id is None:
				wandb_id = wandb.util.generate_id()
			commit_wandb_id(checkpoint_path, wandb_id)
		else:
			wandb_id = wandb.util.generate_id()

		wandb.init(id=wandb_id,
			       resume='allow',
				   project='controllable',
		           entity='controllable', 
		           config=locals(), 
		           job_type='train_bpm_control')

		set_run_path(wandb.run.dir)

	domain_name, task_name = env_name.split('/')

	# env fixes
	frame_skip = 1
	if domain_name == 'point_mass':
		frame_skip = 4

	diagonal = False
	if domain_name == 'manipulator':
		diagonal = True

	include_labels = included_labels[env_name]

	# env
	env_cls = lambda: AnnotatedDMControl(domain_name=domain_name,
										 task_name=task_name,
										 frame_skip=frame_skip,
										 diagonal=diagonal)
	env = env_cls()
	obs = env.reset()
	obs_shape = obs.shape
	n_actions = env.action_space.n
	print('Initialized environment.')

	# model
	if frames_goal:
		shared_encoder = Impala(obs_shape=obs_shape,
								state_size=state_size,
								group_norm=True)
		shared_encoder = AtariEncoder(encoder=shared_encoder)
		obs_encoder = shared_encoder
		task_encoder = shared_encoder
	else:
		obs_encoder = Impala(obs_shape=obs_shape,
							 state_size=state_size,
							 group_norm=True)
		obs_encoder = AtariEncoder(encoder=obs_encoder)

		single_obs_shape = list(obs_shape)
		single_obs_shape[0] = 1

		task_encoder = Impala(obs_shape=single_obs_shape,
							  state_size=state_size,
							  group_norm=True)
		task_encoder = AtariSingleFrameEncoder(encoder=task_encoder,
											   obs_shape=obs_shape)

	model = EncoderLSTMModel(obs_shape=obs_shape,
							 n_actions=n_actions,  
							 device=device,
							 obs_encoder=obs_encoder,
							 task_encoder=task_encoder,
							 obs_rep_size=state_size,
							 task_rep_size=state_size,
							 state_size=state_size,
							 lstm_hidden_dim=lstm_hidden_dim,
							 lstm_layers=lstm_layers,
							 lstm_dropout_p=lstm_dropout_p).to(device=device)
	print('Initialized models.')

	train_glamor(env_cls=env_cls,
			     model=model,
			     policy=policy,
			     planner=planner,
			     final_eps=final_eps,
			     eps_steps=int(eps_steps),
			     n_interactions=int(n_interactions),
			     k=k,
			     lr=lr,
			     policy_trials=policy_trials,
			     clip_p_actions=clip_p_actions,
			     test_policies=test_policies,
			     test_tasks=test_tasks,
			     eval_types=eval_types,
			     include_labels=include_labels,
			     n_label_tasks=n_label_tasks,
			     n_video_tasks=n_video_tasks,
			     n_eval_tasks=n_eval_tasks,
			     device=device,
			     gamma=gamma,
			     n_tasks=n_tasks,
			     replay_ratio=replay_ratio,
			     log_period=int(log_period),
			     snapshot_period=int(snapshot_period),
			     buffer_size=int(buffer_size),
			     min_step_learn=int(min_step_learn),
			     checkpoint_path=checkpoint_path)



