import random
import torch


class UniformK:

    def __init__(self, max_k, min_k=1):
        self.max = max_k
        self.min = min_k

    def sample(self):
        return random.randint(self.min, self.max)

    def action_prior(self, t, batch_size, n_actions):
        if t == 0:
            p_end = 1./(self.max - 1)
        else:
            p_end = 1./(self.max - t)
        other_prob = 1 - p_end
        dist = [other_prob / n_actions] * n_actions + [p_end]

        dist = torch.tensor(dist).unsqueeze(0)
        dist = dist.expand(batch_size, n_actions + 1) + 1e-4
        dist = dist / dist.sum(dim=-1, keepdim=True)
        return torch.log(dist)


class ConstantK:
    def __init__(self, value):
        self.min = value
        self.max = value

    def sample(self):
        return self.min
