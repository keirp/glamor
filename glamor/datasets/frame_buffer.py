import numpy as np
import torch
from glamor.datasets.replay_buffer import ReplayBuffer

from collections import namedtuple

Sample = namedtuple("Sample", ["obs", "obs_k", "actions", "ks"])


def frame_buffer_from_obs_buffer(obs_buffer):
    fb = FrameBuffer(buffer_size=obs_buffer.max_buffer_size,
                     env_cls=obs_buffer.env_cls,
                     k_dist=obs_buffer.k_dist,
                     end_token=obs_buffer.end_token,
                     frames_goal=obs_buffer.frames_goal,
                     action_seq=obs_buffer.action_seq)
    if obs_buffer.current_buffer_size < obs_buffer.max_buffer_size:
        fb.frame_buffer[fb.n_frames - 1:] = obs_buffer.obs_buffer[:, -1]
    else:
        fb.frame_buffer[:obs_buffer.t] = obs_buffer.obs_buffer[:obs_buffer.t, -1]
        fb.frame_buffer[obs_buffer.t + fb.n_frames -
                        1:] = obs_buffer.obs_buffer[obs_buffer.t:, -1]
        fb.frame_buffer[obs_buffer.t:obs_buffer.t + fb.n_frames -
                        1] = obs_buffer.obs_buffer[obs_buffer.t, :-1]

    fb.action_buffer = obs_buffer.action_buffer
    fb.traj_end_buffer = obs_buffer.traj_end_buffer
    fb.t = obs_buffer.t
    fb.t_buffer = obs_buffer.t_buffer
    fb.current_buffer_size = obs_buffer.current_buffer_size

    return fb


def obs_buffer_from_frame_buffer(frame_buffer):
    ob = ReplayBuffer(buffer_size=frame_buffer.max_buffer_size,
                      env_cls=frame_buffer.env_cls,
                      k_dist=frame_buffer.k_dist,
                      end_token=frame_buffer.end_token,
                      frames_goal=frame_buffer.frames_goal,
                      action_seq=frame_buffer.action_seq)
    idxs = np.arange(frame_buffer.current_buffer_size)
    ob.obs_buffer[:frame_buffer.current_buffer_size] = frame_buffer._get_obs(
        idxs)

    ob.action_buffer = frame_buffer.action_buffer
    ob.traj_end_buffer = frame_buffer.traj_end_buffer
    ob.t = frame_buffer.t
    ob.t_buffer = frame_buffer.t_buffer
    ob.current_buffer_size = frame_buffer.current_buffer_size

    return ob


class FrameBuffer:
    """Fully function frame buffer that does not store duplicated
    frames. Used to compress obs_buffer for storage."""

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

        obs_shape = env.observation_space.shape
        self.n_frames = obs_shape[0]
        print(f'obs_shape: {obs_shape}, n_frames: {self.n_frames}')
        self.frame_buffer = np.zeros((buffer_size + self.n_frames - 1, *obs_shape[1:]), dtype=env.observation_space.dtype)

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
            frame_idxs = (np.arange(self.t, self.t + T) + self.n_frames -
                          1) % (self.max_buffer_size + self.n_frames - 1)
            # only most recent frames
            self.frame_buffer[frame_idxs] = stacked_obs[:, -1]
            self.action_buffer[idxs] = actions
            self.traj_end_buffer[idxs] = idxs[-1]
            self.t_buffer[idxs] = np.arange(T)
            self.t = (self.t + T) % self.max_buffer_size

            if self.current_buffer_size < self.max_buffer_size:
                self.current_buffer_size = min(
                    self.current_buffer_size + T, self.max_buffer_size)

    def _get_obs(self, idxs):
        frame_idxs = idxs.copy()
        if self.current_buffer_size < self.max_buffer_size:
            frame_idxs += self.n_frames - 1
        else:
            frame_idxs[frame_idxs >= self.t] += self.n_frames - 1
        lims = [np.arange(t - self.n_frames + 1, t + 1) %
                (self.max_buffer_size + self.n_frames - 1) for t in frame_idxs]
        obs = np.stack([self.frame_buffer[lim] for lim in lims])

        for f in range(self.n_frames - 1):
            blanks = np.where(self.t_buffer[idxs] == f)
            obs[blanks, :self.n_frames - f - 1] = 0

        return obs

    def sample(self, n, idxs=None):
        """Sample n (s, sg, a) pairs and return numpy objects"""
        if idxs is None:
            idxs = np.random.randint(
                low=0, high=self.current_buffer_size, size=(n,))

        ks = np.array([self.k_dist.sample() for _ in range(n)])
        len_remainings = (
            self.traj_end_buffer[idxs] - idxs) % self.max_buffer_size

        ks = np.minimum(ks, len_remainings)

        future_idxs = (idxs + ks) % self.max_buffer_size

        obs = self._get_obs(idxs)

        if self.frames_goal:
            obs_k = self._get_obs(future_idxs)
        else:
            obs_k = self._get_obs(future_idxs)[:, -1][:, np.newaxis]

        # Can't figure out how to do this vectorized...
        if self.action_seq:
            actions_one_hot = np.zeros(
                (n, self.k_dist.max + 1, self.end_token + 2), dtype=self.action_buffer.dtype)

            for i in range(n):
                a_idx = np.arange(idxs[i], idxs[i] + ks[i]
                                  ) % self.max_buffer_size
                actions = np.ones(
                    self.k_dist.max + 1, dtype=self.action_buffer.dtype) * (self.end_token + 1)
                actions[:ks[i]] = self.action_buffer[a_idx]
                actions[ks[i]] = self.end_token

                actions_one_hot[i, np.arange(actions.size), actions] = 1
        else:
            actions_one_hot = np.zeros(
                (n, self.end_token), dtype=self.action_buffer.dtype)
            actions = self.action_buffer[idxs]
            actions_one_hot[np.arange(actions.size), actions] = 1

        return Sample(obs=torch.tensor(obs),
                      obs_k=torch.tensor(obs_k),
                      actions=torch.tensor(actions_one_hot),
                      ks=torch.tensor(ks))
