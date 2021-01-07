import numpy as np
import torch
import torch.nn as nn
import time
from collections import defaultdict

# # https://stackoverflow.com/a/46103129/ @Divakar
def all_idx(idx, axis):
	grid = np.ogrid[tuple(map(slice, idx.shape))]
	grid.insert(axis, idx)
	return tuple(grid)

def onehot_initialization(a, ncols=None):
	if ncols is None:
		ncols = a.max()+1
	out = np.zeros(a.shape + (ncols,), dtype=int)
	out[all_idx(a, axis=2)] = 1
	# print(out.shape)
	out = np.moveaxis(out, -1, 0)
	# print(out.shape)
	return out

def onehot_torch(a, ncols):
	onehot = torch.FloatTensor(a.shape[0], ncols)
	onehot.scatter_(1, a.view(-1, 1), 1)

def onehot_plane(a, ncols, h, w):
	v = onehot_initialization(a, ncols)
	v = v.view(v.shape[0], v.shape[1], 1, 1)
	planes = torch.ones(v.shape[0], v.shape[1], h, w)
	return v * planes

class Timer:

	def __init__(self):
		self.start_time = None
		self.name = None
		self.sub_timers = defaultdict(float)
		self.sub_start_time = None
		self.sub_name = None

	def start(self, name):
		if self.sub_start_time is not None:
			self.sub_timers[self.sub_name] += time.time() - self.sub_start_time
		self._print_timers()
		self.start_time = time.time()
		self.name = name
		self.sub_start_time = None
		self.sub_name = None
		self.sub_timers = defaultdict(float)

	def _print_timers(self):
		if self.sub_start_time is not None:
			for sub_name in self.sub_timers:
				print(f'{sub_name} took {self.sub_timers[sub_name]} seconds.')
		if self.start_time is not None:
			print(f'{self.name} took {time.time() - self.start_time} seconds.')

	def start_sub(self, name):
		if self.sub_start_time is not None:
			self.sub_timers[self.sub_name] += time.time() - self.sub_start_time
		self.sub_start_time = time.time()
		self.sub_name = name

	def end(self):
		self._print_timers()

class Swish(nn.Module):
	
	def forward(self, input_tensor):
		return input_tensor * torch.sigmoid(input_tensor)

def get_lr(optimizer):
	for param_group in optimizer.param_groups:
		return param_group['lr']

def pretty_print(d):
	folders = defaultdict(dict)
	for key in d:
		parts = key.split('/')
		if len(parts) > 1:
			folders[parts[0]][parts[1]] = d[key]
		else:
			folders['Logs'][parts[0]] = d[key]

	for f in folders:
		print(f'{f}:')
		for key in folders[f]:
			print(f'\t{key}: {folders[f][key]}')


