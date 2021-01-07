import torch.nn as nn

class AtariEncoder(nn.Module):	
	"""Divides by 255 to preprocess Atari observations"""

	def __init__(self, encoder):
		super().__init__()
		self.encoder = encoder

	def forward(self, x):
		x = x / 255.
		return self.encoder(x)

class AtariSingleFrameEncoder(nn.Module):
	"""Divides by 255 to preprocess Atari observations.
	Selects only the last frame from the stacked obs."""

	def __init__(self, encoder, obs_shape):
		super().__init__()
		self.encoder = encoder
		self.obs_shape = obs_shape

	def forward(self, x):
		x = x.view(-1, *self.obs_shape)[:, -1].unsqueeze(1)
		x = x / 255.
		return self.encoder(x)