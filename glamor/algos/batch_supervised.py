import torch
from collections import defaultdict

class BatchSupervised:
	"""Calling train(n) samples n batches and trains
	on them. Returns info to log.

	- format_sample: takes a sample from the replay buffer and performs pre-processing
	- model_fn: takes sample and returns model output
	- loss_fn: takes sample + model output and returns a loss to optimize as well as a dict of
	           losses to log.
	- sampler: an object which can sample from some dataset and return a batch.

	- train: Trains the model.

	"""

	def __init__(self, 
		         model, 
		         model_fn, 
		         loss_fn,
		         format_sample,
		         optimizer,
		         sampler,
		         batch_size,
		         clip_param=1):
		self.model = model
		self.model_fn = model_fn
		self.loss_fn = loss_fn
		self.format_sample = format_sample
		self.optimizer = optimizer
		self.sampler = sampler
		self.batch_size = batch_size
		self.clip_param = clip_param

	def train(self, n):
		self.model.train()
		n_batch = 0
		cum_loss = defaultdict(float)

		for i_batch in range(n):
			sample_batch = self.sampler.sample(self.batch_size)

			# self.optimizer.zero_grad()
			for param in self.model.parameters():
				param.grad = None
				
			f_sample = self.format_sample(sample_batch)
			model_res = self.model_fn(f_sample)
			loss, infos = self.loss_fn(f_sample, model_res)
			for key in infos:
				cum_loss[key] += infos[key].item()
			loss.backward()
			torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_param)
			self.optimizer.step()

			n_batch += 1

		loss_logs = {f'train/{key}': cum_loss[key] / n_batch for key in cum_loss}

		return loss_logs