import torch
from torch import nn
import torch.nn.functional as F

from typing import Tuple
import numpy as np

def get_conv_weight_and_bias(
        filter_size: Tuple[int, int],
        num_groups: int,
        input_channels: int,
        output_channels: int,
        bias: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert input_channels % num_groups == 0, "input channels must be divisible by groups number"
    assert output_channels % num_groups == 0, "output channels must be divisible by groups number"
    input_channels = input_channels // num_groups
    weight_matrix = torch.randn(output_channels, input_channels, *filter_size)
    if bias:
        bias_vector = torch.ones(output_channels)
    else:
        bias_vector = None
    return weight_matrix, bias_vector


class MyConvStub:
    def __init__(
            self,
            kernel_size: Tuple[int, int],
            num_groups: int,
            input_channels: int,
            output_channels: int,
            bias: bool,
            stride: int,
            dilation: int,
    ):
        self.weight, self.bias = get_conv_weight_and_bias(kernel_size, num_groups, input_channels, output_channels, bias)
        self.groups = num_groups
        self.stride = stride
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, in_channels, input_height, input_width = x.shape
        output_channels, input_channels , kernel_height, kernel_width = self.weight.shape
                
        output_height = ((input_height - self.dilation * (kernel_height - 1) - 1) // self.stride) + 1
        output_width = ((input_width - self.dilation * (kernel_width - 1) - 1) // self.stride) + 1
        output = torch.zeros((batch_size, output_channels, output_height, output_width))

        for b in range(batch_size):
            for c_out in range(output_channels):
                for h_out in range(output_height):
                    for w_out in range(output_width):
                        output[b, c_out, h_out, w_out] = self._conv_forward(h_out, w_out, c_out, b, x)

        return output

    def _conv_forward(self, h_out, w_out, c_out, b, input_data):
        h_start = h_out * self.stride
        h_end = h_start + self.weight.size(2) * self.dilation
        w_start = w_out * self.stride
        w_end = w_start + self.weight.size(3) * self.dilation

        # if h_end > input_data.size(2) or w_end > input_data.size(3):
        #     return 0.0

        field = input_data[b, :, h_start:h_end:self.dilation, w_start:w_end:self.dilation]
        weight_channel = self.weight[c_out, :, :, :]
    
        field = field * weight_channel
        output = field.sum()

        if self.bias is not None:
            output += self.bias[c_out]

        return output
   
class MyFilterStub:
    def __init__(
            self,
            filter: torch.Tensor,
            input_channels: int,
    ):
        self.weight = filter
        self.input_channels = input_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, in_channels, input_height, input_width = x.shape
        filter_height, filter_width = self.weight.shape
        
        output_height = input_height - filter_height + 1
        output_width = input_width - filter_width + 1
        output = torch.zeros((batch_size, self.input_channels, output_height, output_width), device=x.device)

        for c in range(self.input_channels):
            for b in range(batch_size):
                for h in range(output_height):
                    for w in range(output_width):
                        region = x[b, c, h:h + filter_height, w:w + filter_width]
                        output[b, c, h, w] = (region * self.weight).sum()

        return output
