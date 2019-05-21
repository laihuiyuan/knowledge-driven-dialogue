# -*- coding: UTF-8 -*-

"""
The original version comes from Baidu.com, https://github.com/baidu/knowledge-driven-dialogue
File: source/encoders/rnn_encoder.py
"""

import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence


class RNNEncoder(nn.Module):
    """
    A LSTM recurrent neural network encoder.
    """

    def __init__(self,input_size, hidden_size,
                 highway=None, embedder=None, char_embedder=None,num_layers=1,
                 bidirectional=True, dropout=0.0):
        super(RNNEncoder, self).__init__()

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        rnn_hidden_size = hidden_size // num_directions

        self.dropout = dropout
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn_hidden_size = rnn_hidden_size
        self.highway = highway
        self.embedder = embedder
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.char_embedder = char_embedder


        self.rnn = nn.LSTM(input_size=self.input_size,
                           hidden_size=self.rnn_hidden_size,
                           num_layers=self.num_layers,
                           batch_first=True,
                           dropout=self.dropout if self.num_layers > 1 else 0,
                           bidirectional=self.bidirectional)

    def forward(self, inputs, inputs_c, hidden=None):
        """
        forward
        """
        if isinstance(inputs, tuple):
            inputs, lengths = inputs
        else:
            inputs, lengths = inputs, None

        if self.embedder is not None:
            rnn_inputs = self.embedder(inputs)
        else:
            rnn_inputs = inputs

        if self.char_embedder is not None:
            char_embed=self.char_embedder(inputs_c)
            rnn_inputs=torch.cat([char_embed,rnn_inputs],dim=-1)

        if self.highway is not None:
            rnn_inputs = self.highway(rnn_inputs)

        batch_size = rnn_inputs.size(0)

        if lengths is not None:
            num_valid = lengths.gt(0).int().sum().item()
            sorted_lengths, indices = lengths.sort(descending=True)
            rnn_inputs = rnn_inputs.index_select(0, indices)

            rnn_inputs = pack_padded_sequence(
                rnn_inputs[:num_valid],
                sorted_lengths[:num_valid].tolist(),
                batch_first=True)

            if hidden is not None:
                hidden = (hidden[0].index_select(1, indices)[:, :num_valid],
                          hidden[1].index_select(1, indices)[:, :num_valid])

        outputs, (last_h, last_c) = self.rnn(rnn_inputs, hidden)

        if self.bidirectional:
            last_h = self._bridge_bidirectional_hidden(last_h)
            last_c = self._bridge_bidirectional_hidden(last_c)

        if lengths is not None:
            outputs, _ = pad_packed_sequence(outputs, batch_first=True)

            if num_valid < batch_size:
                zeros = outputs.new_zeros(
                    batch_size - num_valid, outputs.size(1), self.hidden_size)
                outputs = torch.cat([outputs, zeros], dim=0)

                zeros = last_h.new_zeros(
                    self.num_layers, batch_size - num_valid, self.hidden_size)
                last_h = torch.cat([last_h, zeros], dim=1)
                last_c = torch.cat([last_c, zeros], dim=1)

            _, inv_indices = indices.sort()
            outputs = outputs.index_select(0, inv_indices)
            last_h = last_h.index_select(1, inv_indices)
            last_c = last_c.index_select(1, inv_indices)

        return outputs, (last_h, last_c)

    def _bridge_bidirectional_hidden(self, hidden):
        """
        the bidirectional hidden is (num_layers * num_directions, batch_size, hidden_size)
        we need to convert it to (num_layers, batch_size, num_directions * hidden_size)
        """
        num_layers = hidden.size(0) // 2
        _, batch_size, hidden_size = hidden.size()
        return hidden.view(num_layers, 2, batch_size, hidden_size) \
            .transpose(1, 2).contiguous().view(num_layers, batch_size, hidden_size * 2)
