# -*- coding: UTF-8 -*-

"""
File: source/modules/highway.py
"""

import torch
import torch.nn as nn


class Highway(nn.Module):
    def __init__(self, input_dim, num_layers, activation):

        super(Highway, self).__init__()

        self.num_layers = num_layers
        self.gate = nn.ModuleList([nn.Linear(input_dim, input_dim) for _ in range(num_layers)])
        self.linear = nn.ModuleList([nn.Linear(input_dim, input_dim) for _ in range(num_layers)])

        self.act = activation

    def forward(self, inputs):
        
        curr_input = inputs
        for layer in range(self.num_layers):
            linear_part=curr_input
            gate = torch.sigmoid(self.gate[layer](inputs))
            nonlinear_part = self.act(self.linear[layer](inputs))
            curr_input=gate * linear_part + (1 - gate) * nonlinear_part

        return curr_input
