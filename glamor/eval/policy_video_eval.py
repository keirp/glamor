from glamor.eval.base import BaseEval
from glamor.samplers.trajectory_sampler import TrajectorySampler

from PIL import Image
import numpy as np
import wandb

BAR_HEIGHT = 5


class PolicyVideoEval(BaseEval):
    """Generates a video of the policy running in the environment 
    and compares the labels of the terminal states."""

    def __init__(self,
                 env_cls,
                 horizon,
                 policy,
                 tasks,
                 n_trajs,
                 time_to_go=True,
                 eval_freq=30):
        self.sampler = TrajectorySampler(env_cls=env_cls,
                                         policy=policy,
                                         horizon=horizon,
                                         tasks=tasks)
        self.n_trajs = n_trajs
        self.time_to_go = time_to_go
        self.eval_freq = eval_freq
        self._prefix = f'{policy.name}_{tasks.name}_policy_video'

    def _add_time_to_go(self, frame, prog):
        frame = np.pad(frame, [[0, BAR_HEIGHT], [0, 0]], 'constant')
        frame[-BAR_HEIGHT:, :int(frame.shape[1] * prog)] = 255
        return frame

    def eval(self, model):
        print(f'Evaluating {self.prefix}')
        trajs = self.sampler.collect_trajectories(n_interactions=None,
                                                  n_trajs=self.n_trajs)

        frames = []
        for traj in trajs:
            task = traj.task

            for i in range(len(traj.obs)):
                obs = traj.obs[i]
                frame = np.concatenate([obs[-1], task.obs[-1]], axis=1)
                if self.time_to_go:
                    frame = self._add_time_to_go(
                        frame, traj.policy_infos[i].time_to_go)
                frames.append(frame)

            # Freeze the video for 10 frames after a goal is achieved
            for _ in range(10):
                frames.append(frames[-1])

        ann_frames = []
        for i in range(len(frames)):
            frame_image = Image.fromarray(frames[i]).convert('RGBA')
            np_frame = np.array(frame_image)
            np_frame = np.moveaxis(np_frame, -1, 0)
            ann_frames.append(np_frame)

        ann_frames = np.array(ann_frames)

        video = wandb.Video(ann_frames, fps=10, format='mp4')

        logs = {'policy_video': video}

        return logs
