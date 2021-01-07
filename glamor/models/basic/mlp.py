import torch.nn as nn
import torch
from math import floor

class MLP(nn.Module):

	def __init__(self,
				 in_dim,
				 out_dim,
				 hidden=[256, 256, 256],
				 use_layer_norm=True,
		         dropout_p=0,
				 nonlinearity=nn.ReLU):
		super().__init__()

		in_units = [in_dim] + hidden
		out_units = hidden + [out_dim]

		linear_layers = [nn.Linear(i, o) for (i, o) in
			zip(in_units, out_units)]
		if use_layer_norm:
			norms = [nn.LayerNorm(units) for units in hidden]
		else:
			norms = [None for units in hidden]
		sequence = []
		for lin_layer, layer_norm in zip(linear_layers[:-1], norms):
			layer = []
			layer.append(lin_layer)
			if use_layer_norm:
				layer.append(layer_norm)
			if dropout_p > 0:
				layer.append(nn.Dropout(dropout_p))
			layer.append(nonlinearity())
			sequence.extend(layer)
		sequence.append(linear_layers[-1])
		self.linear = torch.nn.Sequential(*sequence)

	def forward(self, input):
		"""Computes the convolution stack on the input; assumes correct shape
		already: [B,C,H,W]."""
		return self.linear(input)