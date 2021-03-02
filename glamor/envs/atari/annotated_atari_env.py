from glamor.envs.atari.atari_env import AtariEnv
from atariari.benchmark.ram_annotations import atari_dict
from rlpyt.envs.base import EnvStep

from collections import namedtuple

EnvInfo = namedtuple(
    "EnvInfo", ["game_score", "traj_done", "labels", "goal_labels"])


class AnnotatedAtariEnv(AtariEnv):

    def __init__(self,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.static_info = self.labels()

    def step(self, action):
        obs, reward, done, info = super().step(action)

        # fix the labels later in GoalWrapper
        updated_info = EnvInfo(game_score=info.game_score,
                               traj_done=info.traj_done,
                               labels=self.static_info,
                               goal_labels=self.static_info)

        return EnvStep(obs, reward, done, updated_info)

    def labels(self):
        ram = self.ale.getRAM()
        label_dict = ram2label(self._game.replace('_', ''), ram)
        TupleCls = namedtuple('AtariLabels', sorted(label_dict))

        return TupleCls(**label_dict)


def ram2label(game, ram):
    if game in atari_dict:
        label_dict = {k: ram[ind] for k, ind in atari_dict[game].items()}
    else:
        assert False, "{} is not currently supported by AARI. It's either not an Atari game or we don't have the ram annotations yet!".format(
            game_name)
    return label_dict
