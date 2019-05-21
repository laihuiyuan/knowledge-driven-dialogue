# -*- coding: UTF-8 -*-

import os
import codecs

import torch
import torch.nn as nn
from torch.autograd import Variable


class CharEmbed(nn.Module):
    def __init__(self, embed_size, char2id, fix_emb=False,
                 oov='<oov>', pad='<pad>', normalize=True):
        super(CharEmbed, self).__init__()
        self.id2word = {i: word for word, i in char2id.items()}
        self.num_vocab, self.embed_size = len(char2id), embed_size
        self.oovid = char2id[oov]
        self.padid = char2id[pad]
        self.embedding = nn.Embedding(self.num_vocab,
                                      self.embed_size,
                                      padding_idx=self.padid)

        self.embedding.weight.data.uniform_(-0.25, 0.25)

        if normalize:
            weight = self.embedding.weight
            norms = weight.data.norm(2, 1)
            if norms.dim() == 1:
                norms = norms.unsqueeze(1)
            weight.data.div_(norms.expand_as(weight.data))

        if fix_emb:
            self.embedding.weight.requires_grad = False

    def forward(self, input_):
        return self.embedding(input_)


class CnnEmbedder(nn.Module):
    def __init__(self, char_file, char_dim, max_chars,
                 filters, activation, projection_dim, use_cuda):
        super(CnnEmbedder, self).__init__()

        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "tanh":
            self.act = nn.Tanh()

        self.emb_dim = 0
        self.use_cuda = use_cuda
        self.max_chars = max_chars
        self.output_dim = projection_dim
        self.n_filters = sum(f[1] for f in filters)
        self.emb_dim += sum(f[1] for f in filters)
        self.projection = nn.Linear(self.emb_dim, self.output_dim, bias=True)

        self.char_lexicon = {}
        self.convolutions = []

        for i, (width, num) in enumerate(filters):
            conv = nn.Conv1d(in_channels=char_dim,
                             out_channels=num,
                             kernel_size=width,
                             bias=True
                             )
            self.convolutions.append(conv)
        self.convolutions = nn.ModuleList(self.convolutions)

        with codecs.open(os.path.join(char_file), 'r', encoding='utf-8') as fin:
            for line in fin:
                tokens = line.strip().split('\t')
                if len(tokens) == 1:
                    tokens.insert(0, '\u3000')
                token, i = tokens
                self.char_lexicon[token] = int(i)
        self.char_embedder = CharEmbed(char_dim, self.char_lexicon, fix_emb=False)

    def forward(self, chars_inp):

        batch_size, seq_len = len(chars_inp),max([len(char) for char in chars_inp])
        chars_inp = chars_inp.view(batch_size * seq_len, -1)

        char_emb = self.char_embedder(Variable(chars_inp).cuda()
                                      if self.use_cuda else Variable(chars_inp))
        char_emb = char_emb.transpose(1, 2)

        convs = []
        for i in range(len(self.convolutions)):
            convolved = self.convolutions[i](char_emb)
            # (batch_size * seq_len, n_filters for this width)
            convolved, _ = torch.max(convolved, dim=-1)
            convolved = self.act(convolved)
            convs.append(convolved)
        char_emb = torch.cat(convs, dim=-1)

        char_emb = char_emb.view(batch_size, -1, self.n_filters)

        return self.projection(char_emb)
