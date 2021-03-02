import torch.nn as nn
import torch
import numpy as np


def get_random_policy(obs_shape, n_actions, epsilon):

    obs_size = np.prod(obs_shape)

    rand_mlp = nn.Sequential(nn.Linear(obs_size, 10),
                             nn.ReLU(),
                             nn.Linear(10, n_actions))

    def policy(obs, t):
        obs = torch.tensor(obs).view(-1).unsqueeze(0).float()
        logits = rand_mlp(obs)
        probs = torch.softmax(logits, dim=-1)[0].tolist()
        probs = [p / sum(probs) for p in probs]
        uniform = 1 / n_actions
        mix_prob = [(1 - epsilon) * p + epsilon * uniform for p in probs]
        return np.random.choice(range(n_actions), p=mix_prob)

    return policy
