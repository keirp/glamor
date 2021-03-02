import torch.nn as nn
import torch
from math import floor


class ConvNet(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels,
                 kernel_sizes,
                 strides,
                 paddings,
                 norm=None,
                 dropout_p=0,
                 nonlinearity=nn.ReLU,
                 final_nonlinearity=nn.ReLU):
        super().__init__()

        in_channels = [in_channels] + hidden_channels
        out_channels = hidden_channels + [out_channels]

        conv_layers = [nn.Conv2d(in_channels=ic, out_channels=oc,
                                 kernel_size=k, stride=s, padding=p) for (ic, oc, k, s, p) in
                       zip(in_channels, out_channels, kernel_sizes, strides, paddings)]
        if norm is not None:
            norms = [norm(c) for c in hidden_channels]
        else:
            norms = [c for c in hidden_channels]
        sequence = []
        for conv_layer, n in zip(conv_layers[:-1], norms):
            layer = []
            layer.append(conv_layer)
            if norm is not None:
                layer.append(n)
            if dropout_p > 0:
                layer.append(nn.Dropout2d(dropout_p))
            layer.append(nonlinearity())
            sequence.extend(layer)
        sequence.append(conv_layers[-1])
        if final_nonlinearity is not None:
            sequence.append(final_nonlinearity())
        self.conv = torch.nn.Sequential(*sequence)

    def forward(self, input):
        """Computes the convolution stack on the input; assumes correct shape
        already: [B,C,H,W]."""
        return self.conv(input)

    def conv_out_size(self, input_shape):
        input_channels, ih, iw = input_shape
        for child in self.conv.children():
            try:
                ih = floor(
                    (ih + 2 * child.padding[0] - child.kernel_size[0]) / child.stride[0]) + 1
                iw = floor(
                    (iw + 2 * child.padding[1] - child.kernel_size[1]) / child.stride[1]) + 1
                input_channels = child.out_channels
            except:
                pass

        return ih * iw * input_channels, (input_channels, ih, iw)
