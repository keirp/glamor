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

## Todo
- Remove dependency on `rlpyt` and allow normal gym environments.

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