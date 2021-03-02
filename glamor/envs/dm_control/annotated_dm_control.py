from dm_control import suite
from glamor.envs.dm_control.pixel_wrapper import Wrapper as PixelWrapper
from rlpyt.envs.base import Env, EnvStep
from rlpyt.spaces.int_box import IntBox
import cv2
import numpy as np
from itertools import product
from collections import namedtuple

EnvInfo = namedtuple("EnvInfo", ["traj_done", "labels", "goal_labels"])

W, H = (104, 80)
ACTIONS = (-1, 0, 1)


class AnnotatedDMControl(Env):

    def __init__(self,
                 domain_name,
                 task_name,
                 n_frames=4,
                 frame_skip=4,
                 diagonal=False,
                 seed=None):
        self.env = suite.load(domain_name=domain_name,
                              task_name=task_name, task_kwargs=dict(random=seed))
        self.env = PixelWrapper(self.env, pixels_only=False, render_kwargs={'camera_id': 0,
                                                                            'width': W,
                                                                            'height': H})
        action_spec = self.env.action_spec()
        n_dims = action_spec.shape[0]

        if not diagonal:
            self.action_map = tuple(product(*([ACTIONS] * n_dims)))
        else:
            self.action_map = []
            for dim in range(n_dims):
                action = [0] * n_dims
                action[dim] = 1
                self.action_map.append(action)
                action = [0] * n_dims
                action[dim] = -1
                self.action_map.append(action)
            self.action_map.append([0] * n_dims)
            self.action_map = tuple(self.action_map)
        n_actions = len(self.action_map)

        obs_shape = (n_frames, H, W)
        self.frame_skip = frame_skip
        self._action_space = IntBox(low=0, high=n_actions)
        self._observation_space = IntBox(
            low=0, high=255, shape=obs_shape, dtype="uint8")

        print(f'Starting env with obs_space={obs_shape} and {n_actions} actions.')

        self._obs = np.zeros(shape=obs_shape, dtype="uint8")
        self.last_obs = None

        self.static_labels = None

    def reset(self):
        self._obs[:] = 0
        env_step = self.env.reset()
        self._update_obs(env_step.observation['pixels'])
        self.last_obs = env_step.observation
        return self.get_obs()

    def step(self, action):
        cont_action = np.array(self.action_map[action]).astype(float)
        done = False
        for _ in range(self.frame_skip - 1):
            env_step = self.env.step(cont_action, blind=True)
            if env_step.last():
                done = True
                break
        if not done:
            env_step = self.env.step(cont_action)
        done = env_step.last()
        self._update_obs(env_step.observation['pixels'])
        self.last_obs = env_step.observation

        if self.static_labels is None:
            self.static_labels = self.labels()

        info = EnvInfo(traj_done=done, labels=self.static_labels,
                       goal_labels=self.static_labels)
        return EnvStep(self.get_obs(), 0, done, info)

    def labels(self):
        labels = {}
        for key in self.last_obs.keys():
            if key != 'pixels':
                values = self.last_obs[key].flatten()
                dims = values.shape[0]
                for i in range(dims):
                    labels[f'{key}_{i}'] = values[i]
        TupleCls = namedtuple('ControlSuiteLabels', sorted(labels))
        return TupleCls(**labels)

    def render(self):
        pass

    def get_obs(self):
        return self._obs.copy()

    def _update_obs(self, latest_obs):
        # latest_obs = np.moveaxis(latest_obs, -1, 0)
        # convert latest obs to greyscale
        gray_image = cv2.cvtColor(latest_obs, cv2.COLOR_BGR2GRAY)
        # print(gray_image.shape, self._obs.shape)
        # gray_image = cv2.resize(gray_image[:, 4:-4], (W, H), cv2.INTER_NEAREST)
        self._obs = np.concatenate([self._obs[1:], gray_image[np.newaxis]])
