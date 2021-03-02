from glamor.samplers.base import BaseSampler
from collections import namedtuple
from random import randint

Trajectory = namedtuple(
    "Trajectory", ["obs", "actions", "infos", "task", "policy_infos"])


class TrajectorySampler(BaseSampler):
    """Collects full trajectories. If there is an unfinished trajectory
    (t < horizon or not `done`), don't return those transitions.

    Takes an iterable `goals`.	"""

    def __init__(self,
                 env_cls,
                 policy,
                 horizon,
                 tasks,
                 lazy_labels=False,
                 random_horizon=False
                 ):
        self.env_cls = env_cls
        self.env = self.env_cls()
        self.policy = policy
        self.horizon = horizon
        self.tasks = tasks
        self.lazy_labels = lazy_labels
        self.random_horizon = random_horizon

    def collect_trajectories(self, n_interactions, n_trajs=None):
        """Collect at most n_interactions. If n_trajs is not None,
        collecta at most n_trajs trajectories."""

        if n_interactions is not None:
            print(f'Using {self.policy.name} to gather {n_interactions} interactions.')

        trajs = []

        n_gathered = 0

        self.policy.reset()

        obs_ = []
        actions_ = []
        infos_ = []
        policy_infos_ = []
        t = 0

        env = self.env

        end_token = env.action_space.n

        task_itr = iter(self.tasks)
        task = next(task_itr)

        obs = env.reset()

        if self.random_horizon:
            horizon = randint(1, self.horizon)
        else:
            horizon = self.horizon

        while n_interactions is None or n_gathered < n_interactions:
            obs_.append(obs)

            action, policy_info = self.policy.sample(obs, task, t)

            policy_infos_.append(policy_info)
            if action == end_token:
                done = True
                if self.lazy_labels:
                    info = infos_[-1]._replace(labels=env.labels())
                    infos_[-1] = info
            else:
                actions_.append(action)

                obs, reward, done, info = env.step(action)

                t += 1
                if self.lazy_labels and (t == horizon or done):
                    info = info._replace(labels=env.labels())
                infos_.append(info)

            n_gathered += 1

            if t == horizon or done:
                trajs.append(Trajectory(obs=obs_,
                                        actions=actions_,
                                        infos=infos_,
                                        task=task,
                                        policy_infos=policy_infos_))
                t = 0
                obs_ = []
                actions_ = []
                infos_ = []
                policy_infos_ = []

                if n_trajs is not None and len(trajs) == n_trajs:
                    break

                obs = env.reset()
                self.policy.reset()

                if self.random_horizon:
                    horizon = randint(1, self.horizon)
                else:
                    horizon = self.horizon

                try:
                    task = next(task_itr)
                except StopIteration:
                    break

        return trajs
