# Goal-Conditioned Latent Action MOdels for RL (GLAMOR)

## Setup

Create the Conda environment:
```
conda env create -f environment.yaml
```

Install additional dependencies:
```
pip install -e . dependencies/rlpyt git+git://github.com/mila-iqia/atari-representation-learning.git
```

## Example

To train GLAMOR on Atari with default hyperparameters (same as used in the paper), use:

```
python main.py train_glamor_atari --use_wandb=False --run_path='runs/'
```

There is also an included notebook that can be used to train GLAMOR on a GridWorld task.

## Structure

- glamor
  - algos
    - batch_supervised.py (nn training loop)
    - batch_train_glamor.py (main algo logic)
  - datasets
    - frame_buffer.py (replay buffer that only stores each frame once in memory)
    - k_dist.py (code for sampling sequence lengths during training)
    - replay_buffer.py (uniform replay buffer)
  - envs (contains Atari, DM Control Suite, and GridWorld envs)
  - eval
    - label_compare_eval.py (evaluates policy in an env and returns statistics about achieved goals based on labels)
    - policy_video_eval.py (records videos of policies)
  - models
    - atari (pre-processing for Atari models)
    - basic (basic nn blocks)
    - encoder_lstm_model.py (main model class)
  - planner (contains the planning code)
  - policies (different policies like random, open and closed loop policies based on a plan, and eps-greedy)
  - samplers (code for sampling trajectories from the environment using a policy)
  - tasks (code for generating and sampling from task distributions)
  - train
    - scripts.py (main entry point, contains argument definitions)

## Todo
- Remove dependency on `rlpyt` and support normal gym environments.
- Rewrite replay buffers to support non-visual goals.
- Multi-processing for trajectory collection.

## Bibtex

```
@article{paster2020planning,
title={Planning from Pixels using Inverse Dynamics Models}, 
author={Keiran Paster and Sheila A. McIlraith and Jimmy Ba},
year={2020},
eprint={2012.02419},
archivePrefix={arXiv},
primaryClass={cs.LG}
}
```