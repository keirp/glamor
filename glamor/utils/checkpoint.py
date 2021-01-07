from collections import namedtuple
import dill
import os

STATE_FILE = lambda x: f'{x}/checkpoint.dl'
ID_FILE = lambda x: f'{x}/id.dl'

"""Records all state information about training. Only checkpoints at
log times."""
Checkpoint = namedtuple("Checkpoint", ["replay_buffer",
                                       "optimizer_params", 
                                       "model_params",
                                       "total_steps",
                                       "last_snapshot",
                                       ])

def restore_state(checkpoint_path):
	if os.path.exists(STATE_FILE(checkpoint_path)):
		with open(STATE_FILE(checkpoint_path), 'rb') as file:
			chk = dill.load(file)
		return chk
	else:
		return None

def commit_state(checkpoint_path, 
	             model,
			     optimizer,
			     replay_buffer,
			     total_steps,
			     last_snapshot):
	state = Checkpoint(replay_buffer=replay_buffer,
		               optimizer_params=optimizer.state_dict(),
		               model_params=model.state_dict(),
		               total_steps=total_steps,
		               last_snapshot=last_snapshot)
	with open(STATE_FILE(checkpoint_path), 'wb') as file:
		print('Starting to dump state')
		dill.dump(state, file, protocol=-1)
		print('DUMPED LATEST STATE')

def restore_wandb_id(checkpoint_path):
	if os.path.exists(ID_FILE(checkpoint_path)):
		with open(ID_FILE(checkpoint_path), 'rb') as file:
			chk = dill.load(file)
		print('RESUMING!')
		return chk
	else:
		print('NO CHECKPOINT FOUND, STARTING NEW RUN.')
		return None

def commit_wandb_id(checkpoint_path, id):
	with open(ID_FILE(checkpoint_path), 'wb') as file:
		dill.dump(id, file, protocol=4)