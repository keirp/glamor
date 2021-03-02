from glamor.eval.base import BaseEval
from glamor.samplers.trajectory_sampler import TrajectorySampler
from glamor.tasks.goals.terminal_state_goal_dist import ListTerminalStateGoalDist
from glamor.policies.random import RandomPolicy

from PIL import Image
import wandb
from math import sqrt
from collections import defaultdict


class LabelCompareEval(BaseEval):
    """Generates a video of the policy running in the environment 
    and compares the labels of the terminal states."""

    def __init__(self,
                 env_cls,
                 horizon,
                 policy,
                 tasks,
                 n_trajs,
                 include_labels=None,
                 n_bound_samples=500,
                 eval_freq=30,
                 output_trajs=False):
        self.sampler = TrajectorySampler(env_cls=env_cls,
                                         policy=policy,
                                         horizon=horizon,
                                         tasks=tasks,
                                         lazy_labels=True)
        self.n_trajs = n_trajs
        self.eval_freq = eval_freq
        self._prefix = f'{policy.name}_{tasks.name}_label_diff'

        self.include_labels = include_labels
        self.output_trajs = output_trajs

        self.cutoffs = self._get_label_cutoffs(
            env_cls, horizon, n_bound_samples)

    def eval(self, model):
        print(f'Evaluating {self.prefix}')
        trajs = self.sampler.collect_trajectories(n_interactions=None,
                                                  n_trajs=self.n_trajs)

        label_diffs = []
        for traj in trajs:
            task = traj.task

            traj_labels = traj.infos[-1].labels
            task_labels = task.info.labels
            label_diffs.append(self.compare_labels(traj_labels, task_labels))

        avg_label_diffs = {}
        avg_value = 0
        for key in label_diffs[0]:
            value = 0
            for di in label_diffs:
                value += di[key]
            value /= len(label_diffs)
            avg_label_diffs[key] = value
            avg_value += value

        avg_value /= len(label_diffs[0].keys())
        avg_label_diffs['avg_diff'] = avg_value

        avg_label_diffs['avg_pos_diff'] = self._get_euclidean_average(
            avg_label_diffs)

        labels_achieved = self._get_prop_achieved(label_diffs)
        for key in labels_achieved:
            avg_label_diffs[f'achieved_{key}'] = labels_achieved[key]

        total_achieved = self._get_total_achieved(
            label_diffs, include=self.include_labels)
        avg_label_diffs['total_achieved'] = total_achieved

        if self.output_trajs:
            avg_label_diffs['trajs'] = trajs

        return avg_label_diffs

    def _get_label_cutoffs(self, env_cls, horizon, n_bound_samples):
        """Gathers N trajectories with a random policy. Records 10% distances
        for each dimension."""
        env = env_cls()
        goals = ListTerminalStateGoalDist(env_cls=env_cls,
                                          horizon=horizon,
                                          policy=RandomPolicy(
                                              action_space=env.action_space),
                                          n_goals=n_bound_samples,
                                          include_terminal=True,
                                          random_horizon=True)

        mins = {}
        maxs = {}

        for goal in goals:
            labels = goal.info.labels
            label_dict = labels._asdict()
            for key in label_dict:
                v = label_dict[key]
                if key in mins:
                    mins[key] = min(v, mins[key])
                    maxs[key] = max(v, maxs[key])
                else:
                    mins[key] = v
                    maxs[key] = v

        cutoffs = {}
        for key in mins:
            # avoid 0
            range_ = maxs[key] - mins[key]
            if range_ < 0.01:
                range_ = 1
            cutoffs[key] = range_ * 0.1

        print(mins)
        print(maxs)
        print(cutoffs)

        return cutoffs

    def _get_prop_achieved(self, label_diffs):
        achieved = {}
        for key in label_diffs[0]:
            value = 0
            for di in label_diffs:
                if di[key] < self.cutoffs[key]:
                    value += 1
            value /= len(label_diffs)
            achieved[key] = value

        return achieved

    def _get_total_achieved(self, label_diffs, include=None):
        """Calculates the intersection of achieved labels"""
        achieved = 0
        if include is None:
            include = []
            for key in label_diffs[0]:
                include.append(key)
        for di in label_diffs:
            a = True
            for key in di:
                if key in include and di[key] >= self.cutoffs[key]:
                    a = False
                    break
            if a:
                achieved += 1
        achieved /= len(label_diffs)
        return achieved

    def _get_euclidean_average(self, labels):
        included_pos = set()
        total_dist = 0
        for key in labels:
            split_key = key.split('_')
            suffix = split_key[-1]
            entity_name = '_'.join(split_key[:-1])
            if suffix == 'x' or suffix == 'y' and not entity_name in included_pos:
                x_pos, y_pos = 0, 0
                x_label = f'{entity_name}_x'
                y_label = f'{entity_name}_y'

                if x_label in labels:
                    x_pos = labels[x_label]
                if y_label in labels:
                    y_pos = labels[y_label]

                dist = sqrt(x_pos ** 2 + y_pos ** 2)
                total_dist += dist
                included_pos.add(entity_name)
        if len(included_pos) == 0:
            return 0
        return total_dist / len(included_pos)

    def _control_metric(self, labels):
        total_dist = 0

    def compare_labels(self, traj_labels, task_labels):
        """Returns a dict of the absolute values between label pairs."""
        traj_labels = traj_labels._asdict()
        task_labels = task_labels._asdict()

        label_diffs = {}

        for key in traj_labels:
            label_diffs[key] = abs(
                float(traj_labels[key]) - float(task_labels[key]))

        return label_diffs
