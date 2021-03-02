from glamor.tasks.list_task_dist import ListTaskDist, RandomListTaskDist
from glamor.tasks.base import NoneTaskDist
from glamor.samplers.trajectory_sampler import TrajectorySampler
from collections import namedtuple

from tqdm import tqdm
import numpy as np
from random import randint

Goal = namedtuple('Goal', ['obs', 'info'])


def label_features(l, include_fn=None):
    feats = []
    for key in l:
        if include_fn is None or include_fn(key):
            feats.append(float(l[key]))
    return np.array(feats)


def avg_distance(feat_matrix, candidate_feat, exclude_idx):
    """Gets the average L2 distance between a candidate
    and points in feat_matrix.

    - feat_matrix: [n, d]
    - candidate_feat: [n]
    - exclude_idx: [0, n)"""
    candidate_feat = candidate_feat.reshape(1, -1)
    diffs = (candidate_feat - feat_matrix) ** 2
    diffs[exclude_idx] = np.inf
    return np.sqrt(np.sum(diffs, axis=-1)).min(axis=0)


def propose_goal(feat_matrix, goal, include_fn=None):
    n = feat_matrix.shape[0]
    old_idx = randint(0, n - 1)

    goal_feat = label_features(goal.info.labels._asdict(), include_fn)

    old_fit = avg_distance(feat_matrix, feat_matrix[old_idx], old_idx)
    goal_fit = avg_distance(feat_matrix, goal_feat, old_idx)

    if goal_fit > old_fit:
        feat_matrix[old_idx] = goal_feat
        return True, old_idx
    return False, None


def generate_goals(env_cls, horizon, policy, n_goals, n_samples, include_fn=None, random_horizon=False):
    tasks = NoneTaskDist()
    sampler = TrajectorySampler(env_cls=env_cls,
                                policy=policy,
                                horizon=horizon,
                                tasks=tasks,
                                lazy_labels=True,
                                random_horizon=random_horizon)

    # gather example label to figure out d
    traj = sampler.collect_trajectories(n_interactions=None,
                                        n_trajs=1)[0]
    labels = traj.infos[-1].labels
    feats = label_features(labels._asdict(), include_fn)

    feat_matrix = np.zeros([n_goals, feats.shape[0]])

    goals_collected = 0
    goals = []

    for _ in tqdm(range(n_samples)):
        traj = sampler.collect_trajectories(n_interactions=None,
                                            n_trajs=1)[0]
        if len(traj.obs) == horizon:
            goal = Goal(obs=traj.obs[-1], info=traj.infos[-1])
            if len(goals) < n_goals:
                goals.append(goal)
                feats = label_features(goal.info.labels._asdict(), include_fn)
                feat_matrix[len(goals) - 1] = feats
            else:
                accept, idx = propose_goal(feat_matrix, goal, include_fn)
                if accept:
                    goals[idx] = goal

    return goals


class ListDiverseLabelGoalDist(ListTaskDist):
    def __init__(self, env_cls, horizon, policy, n_goals, n_samples, include_fn=None, random_horizon=False):
        tasks = generate_goals(env_cls, horizon, policy,
                               n_goals, n_samples, include_fn, random_horizon)
        super().__init__(tasks)
        self._name = 'list_diverse_label'


class RandomDiverseLabelGoalDist(RandomListTaskDist):
    def __init__(self, env_cls, horizon, policy, n_goals, n_samples, include_fn=None, random_horizon=False):
        tasks = generate_goals(env_cls, horizon, policy,
                               n_goals, n_samples, include_fn, random_horizon)
        super().__init__(tasks)
        self._name = 'random_diverse_label'
