from glamor.datasets.replay_buffer import ReplayBuffer
from glamor.datasets.frame_buffer import FrameBuffer
from glamor.algos.batch_supervised import BatchSupervised
from glamor.samplers.trajectory_sampler import TrajectorySampler
from glamor.utils.checkpoint import restore_state, commit_state
from glamor.utils.utils import pretty_print

import torch
import torch.optim as optim
import torch.nn
import wandb

import numpy as np
from tqdm import tqdm
from time import time
import os

from glamor.utils.logging import wandb_is_enabled, run_path


class BatchTrainGLAMOR:
    """Trains a GLAMOR model.

    - env_cls: function that returns new environment instances
    - horizon: max steps in a trajectory before resetting
    - k_dist: distribution of k (length of trajectory segments on which to train)
    - n_interactions: total interactions during training
    - replay_ratio: how many times to use a data point before sampling a new one from the env
    - policy: policy to use when gathering new samples
    - task_dist: training task (goal) distribution
    - model: model object
    - evals: a list of evaluators which can test the model and return things to log
    - batch_size: sgd batch size
    - log_period: log every log_period interactions
    - snapshot_period: save a snapshot of the model every snapshot_period interactions
    - buffer_size: replay buffer size
    - min_step_learn: take this many steps before updating the model in the beginning
    - lr: sgd learning rate
    - clip_param: gradient clipping param
    - device: cuda/cpu
    - checkpoint_path: path where the checkpoint is stored, if any
    """

    def __init__(self,
                 env_cls,
                 horizon,
                 k_dist,
                 n_interactions,
                 replay_ratio,  # How many times to use a data point before getting a new one
                 policy,
                 task_dist,
                 model,
                 evals,
                 batch_size,
                 log_period=int(1e5),  # log every 1e5 interactions
                 snapshot_period=int(1e6),
                 buffer_size=int(1e6),
                 min_step_learn=int(1e4),
                 lr=1e-4,
                 clip_param=1,
                 device='cuda',
                 frame_buffer=True,
                 checkpoint_path=None,
                 ):

        if wandb_is_enabled():
            wandb.save('*.pt')

        self.n_interactions = n_interactions

        r_prime = replay_ratio * (float(horizon) / float(batch_size))

        if r_prime > 1:
            self.n_trajs = 1
            self.n_batches = round(r_prime)
        else:
            self.n_trajs = round(1. / r_prime)
            self.n_batches = 1

        effective_r = float(self.n_batches * batch_size) / \
            float(self.n_trajs * horizon)

        print(f'Doing {self.n_batches} batches to {self.n_trajs} trajs.')
        print(f'Effective replay ratio: {effective_r}')
        self.model = model
        self.log_period = log_period
        self.snapshot_period = snapshot_period
        self.min_step_learn = min_step_learn
        self.evals = evals
        self.device = device

        self.action_space = env_cls().action_space
        self.n_actions = self.action_space.n

        optimizer = optim.AdamW(self.model.parameters(), lr=lr)

        if frame_buffer:
            self.replay_buffer = FrameBuffer(buffer_size=buffer_size,
                                             env_cls=env_cls,
                                             k_dist=k_dist,
                                             end_token=self.n_actions,
                                             action_seq=True)
        else:
            self.replay_buffer = ReplayBuffer(buffer_size=buffer_size,
                                              env_cls=env_cls,
                                              k_dist=k_dist,
                                              end_token=self.n_actions,
                                              action_seq=True)

        self.trainer = BatchSupervised(model=self.model,
                                       model_fn=self.model_fn,
                                       loss_fn=self.loss_fn,
                                       format_sample=self.format_sample,
                                       optimizer=optimizer,
                                       sampler=self.replay_buffer,
                                       batch_size=batch_size,
                                       clip_param=clip_param)

        self.policy = policy
        self.env_sampler = TrajectorySampler(env_cls=env_cls,
                                             policy=self.policy,
                                             horizon=horizon,
                                             tasks=task_dist,
                                             lazy_labels=False)

        self.ce_loss = torch.nn.NLLLoss(
            reduction='mean', ignore_index=self.n_actions + 1)

        self.total_steps = 0
        self.last_snapshot = 0

        self.checkpoint_path = checkpoint_path

        if self.checkpoint_path is not None:
            state = restore_state(self.checkpoint_path)
            if state is not None:
                self.model.load_state_dict(state.model_params)
                self.trainer.optimizer.load_state_dict(state.optimizer_params)
                self.replay_buffer = state.replay_buffer
                self.trainer.sampler = self.replay_buffer
                self.total_steps = state.total_steps
                print(state.total_steps)
                self.last_snapshot = state.last_snapshot

        if wandb_is_enabled():
            wandb.watch(self.model)

    def model_fn(self, f_sample):
        obs_0, obs_k, actions = f_sample
        actions, baseline_actions = self.model.pred_actions(
            obs_0, obs_k, actions)
        return actions, baseline_actions

    def loss_fn(self, f_sample, model_res):
        actions, baseline_actions = model_res
        _, _, actions_target = f_sample
        actions_target = actions_target.view(-1, self.n_actions + 2)
        actions_target = torch.argmax(actions_target, dim=1)
        actions = actions.reshape(-1, self.n_actions + 2)
        baseline_actions = baseline_actions.reshape(-1, self.n_actions + 2)

        actions_loss = self.ce_loss(actions, actions_target)
        baseline_actions_loss = self.ce_loss(baseline_actions, actions_target)

        loss = actions_loss + baseline_actions_loss
        return loss, {'actions_ce_loss': actions_loss,
                      'baseline_actions_loss': baseline_actions_loss,
                      'ce_loss': loss,
                      'main_loss': loss}

    def format_sample(self, sample_batch):
        batch_size = sample_batch.actions.shape[0]
        actions = sample_batch.actions.float().to(device=self.device)
        obs_0 = sample_batch.obs.float().to(device=self.device).view(batch_size, -1)
        obs_k = sample_batch.obs_k.float().to(device=self.device).view(batch_size, -1)

        return obs_0, obs_k, actions

    def train(self):

        last_log = 0

        prog = tqdm(total=self.log_period)
        last_log_time = time()

        logs = {}
        logs['cum_steps'] = 0
        if self.total_steps == 0:
            self.eval(logs, snapshot=True)
        else:
            last_log = self.total_steps

        while self.total_steps < self.n_interactions:
            self.policy.update(self.total_steps)
            self.model.eval()
            trajs = self.env_sampler.collect_trajectories(
                n_interactions=None, n_trajs=self.n_trajs)
            self.replay_buffer.append_trajs(trajs)
            new_steps = sum([len(traj.obs) for traj in trajs])
            prog.update(new_steps)
            self.total_steps += new_steps
            if len(self.replay_buffer) > self.min_step_learn:
                logs = self.trainer.train(self.n_batches)
            else:
                logs = {}
            if self.total_steps - last_log > self.log_period:
                last_log = self.total_steps
                prog.close()
                time_since_log = time() - last_log_time
                step_per_sec = self.log_period / time_since_log
                logs['step_per_sec'] = step_per_sec
                logs['cum_steps'] = self.total_steps
                last_log_time = time()

                snapshot = self.total_steps - self.last_snapshot > self.snapshot_period
                if snapshot:
                    self.last_snapshot = self.total_steps

                self.eval(logs, snapshot=snapshot)
                prog = tqdm(total=self.log_period)

        # Log after the training is finished too.
        time_since_log = time() - last_log_time
        step_per_sec = self.log_period / time_since_log
        logs['step_per_sec'] = step_per_sec
        logs['cum_steps'] = self.total_steps
        self.eval(logs, snapshot=True)

    def eval(self, logs, snapshot=True):
        print('Beginning logging...')
        eval_start_time = time()

        self.model.eval()
        if self.evals is not None:
            for eval_ in self.evals:
                l = eval_.eval(self.model)
                prefix = eval_.prefix
                eval_logs = {f'{prefix}/{key}': value for key, value in l.items()}
                logs.update(eval_logs)
        logs['replay_size'] = len(self.replay_buffer)
        if self.policy.name == 'eps_greedy':
            logs['agent_eps'] = self.policy.probs[1]
        eval_duration = time() - eval_start_time
        logs['eval_time'] = eval_duration

        if wandb_is_enabled():
            wandb.log(logs)
        pretty_print(logs)

        if self.checkpoint_path is not None:
            commit_state(checkpoint_path=self.checkpoint_path,
                         model=self.model,
                         optimizer=self.trainer.optimizer,
                         replay_buffer=self.replay_buffer,
                         total_steps=self.total_steps,
                         last_snapshot=self.last_snapshot)

        # save model
        if snapshot:
            torch.save(self.model.state_dict(), os.path.join(run_path(), f'model_{logs["cum_steps"]}.pt'))
