# -*- coding: UTF-8 -*-

"""
The original version comes from Baidu.com, https://github.com/baidu/knowledge-driven-dialogue
File: source/decoders/state.py
"""


class DecoderState(object):
    """
    State of Decoder.
    """

    def __init__(self, hidden=None, **kwargs):
        """
        hidden: Tensor(num_layers, batch_size, hidden_size)
        """
        if hidden is not None:
            self.hidden = hidden
        for k, v in kwargs.items():
            if v is not None:
                self.__setattr__(k, v)

    def __getattr__(self, name):
        return self.__dict__.get(name)

    def get_batch_size(self):
        """
        get_batch_size
        """
        if self.hidden is not None:
            return self.hidden[0].size(1)
        else:
            return next(iter(self.__dict__.values())).size(0)

    def size(self):
        """
        size
        """
        sizes = {k: v.size() for k, v in self.__dict__.items()}
        return sizes

    def slice_select(self, stop):
        """
        slice_select
        """
        kwargs = {}
        for k, v in self.__dict__.items():
            if k == "hidden":
                kwargs[k] = (v[0][:, :stop].clone(),v[1][:, :stop].clone())
            else:
                kwargs[k] = v[:stop]
        return DecoderState(**kwargs)

    def index_select(self, indices):
        """
        index_select
        """
        kwargs = {}
        for k, v in self.__dict__.items():
            if k == 'hidden':
                kwargs[k] = (v[0].index_select(1, indices),v[1].index_select(1, indices))
            else:
                kwargs[k] = v.index_select(0, indices)
        return DecoderState(**kwargs)

    def mask_select(self, mask):
        """
        mask_select
        """
        kwargs = {}
        for k, v in self.__dict__.items():
            if k == "hidden":
                kwargs[k] = (v[0][:, mask],v[1][:, mask])
            else:
                kwargs[k] = v[mask]
        return DecoderState(**kwargs)

    def _inflate_tensor(self, X, times):
        """
        inflate X from shape (batch_size, ...) to shape (batch_size*times, ...)
        for first decoding of beam search
        """
        sizes = X.size()

        if X.dim() == 1:
            X = X.unsqueeze(1)

        repeat_times = [1] * X.dim()
        repeat_times[1] = times
        X = X.repeat(*repeat_times).view(-1, *sizes[1:])
        return X

    def inflate(self, times):
        """
        inflate
        """
        kwargs = {}
        for k, v in self.__dict__.items():
            if k == "hidden":
                num_layers, batch_size, _ = v[0].size()
                kwargs[k] = (v[0].repeat(1, 1, times).view(num_layers, batch_size * times, -1),
                             v[1].repeat(1, 1, times).view(num_layers, batch_size * times, -1))
            else:
                kwargs[k] = self._inflate_tensor(v, times)
        return DecoderState(**kwargs)
