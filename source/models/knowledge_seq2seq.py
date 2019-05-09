# -*- coding: UTF-8 -*-

"""
The original version comes from Baidu.com, https://github.com/baidu/knowledge-driven-dialogue
File: source/models/knowledge_seq2seq.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from source.utils.misc import Pack
from source.utils.metrics import accuracy
from source.utils.criterions import NLLLoss
from source.modules.embedder import Embedder
from source.modules.attention import Attention
from source.models.base_model import BaseModel
from source.modules.encoders.rnn_encoder import RNNEncoder
from source.modules.decoders.rnn_decoder import RNNDecoder


class KnowledgeSeq2Seq(BaseModel):
    """
    KnowledgeSeq2Seq
    """

    def __init__(self, src_vocab_size, tgt_vocab_size, embed_size, hidden_size, padding_idx=None,
                 num_layers=1, bidirectional=True, attn_mode="mlp", attn_hidden_size=None,
                 with_bridge=False, tie_embedding=False, dropout=0.0, use_gpu=False, use_bow=False,
                 use_kd=False, use_dssm=False, use_posterior=False, weight_control=False,
                 use_pg=False, use_gs=False, concat=False, pretrain_epoch=0):
        super(KnowledgeSeq2Seq, self).__init__()

        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.padding_idx = padding_idx
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.attn_mode = attn_mode
        self.attn_hidden_size = attn_hidden_size
        self.with_bridge = with_bridge
        self.tie_embedding = tie_embedding
        self.dropout = dropout
        self.use_gpu = use_gpu
        self.use_bow = use_bow
        self.use_dssm = use_dssm
        self.weight_control = weight_control
        self.use_kd = use_kd
        self.use_pg = use_pg
        self.use_gs = use_gs
        self.use_posterior = use_posterior
        self.pretrain_epoch = pretrain_epoch
        self.baseline = 0

        enc_embedder = Embedder(num_embeddings=self.src_vocab_size,
                                embedding_dim=self.embed_size,
                                padding_idx=self.padding_idx)

        if self.tie_embedding:
            assert self.src_vocab_size == self.tgt_vocab_size
            dec_embedder = enc_embedder
            kng_embedder = enc_embedder
        else:
            dec_embedder = Embedder(num_embeddings=self.tgt_vocab_size,
                                    embedding_dim=self.embed_size,
                                    padding_idx=self.padding_idx)
            kng_embedder = Embedder(num_embeddings=self.tgt_vocab_size,
                                    embedding_dim=self.embed_size,
                                    padding_idx=self.padding_idx)

        self.encoder = RNNEncoder(input_size=self.embed_size, hidden_size=self.hidden_size,
                                  embedder=enc_embedder, num_layers=self.num_layers,
                                  bidirectional=self.bidirectional, dropout=self.dropout)

        self.kng_encoder = RNNEncoder(input_size=self.embed_size,
                                      hidden_size=self.hidden_size,
                                      embedder=kng_embedder,
                                      num_layers=self.num_layers,
                                      bidirectional=self.bidirectional,
                                      dropout=self.dropout)

        self.tgt_encoder = RNNEncoder(input_size=self.embed_size,
                                      hidden_size=self.hidden_size,
                                      embedder=kng_embedder,
                                      num_layers=self.num_layers,
                                      bidirectional=self.bidirectional,
                                      dropout=self.dropout)

        self.decoder = RNNDecoder(input_size=self.embed_size, hidden_size=self.hidden_size,
                                  output_size=self.tgt_vocab_size, embedder=dec_embedder,
                                  num_layers=self.num_layers, attn_mode=self.attn_mode,
                                  memory_size=self.hidden_size, feature_size=None,
                                  dropout=self.dropout, concat=concat)

        self.pri_attention = Attention(query_size=self.hidden_size,
                                       memory_size=self.hidden_size,
                                       hidden_size=self.hidden_size,
                                       mode="dot")

        self.pos_attention = Attention(query_size=self.hidden_size,
                                       memory_size=self.hidden_size,
                                       hidden_size=self.hidden_size,
                                       mode="dot")

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        self.log_softmax = nn.LogSoftmax(dim=-1)

        # if self.with_bridge:
        #     self.bridge = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size), nn.Tanh())

        if self.use_bow:
            self.bow_output_layer = nn.Sequential(
                nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size),
                nn.Tanh(),
                nn.Linear(in_features=self.hidden_size, out_features=self.tgt_vocab_size),
                nn.LogSoftmax(dim=-1))

        if self.use_kd:
            self.knowledge_dropout = nn.Dropout()

        if self.padding_idx is not None:
            self.weight = torch.ones(self.tgt_vocab_size)
            self.weight[self.padding_idx] = 0
        else:
            self.weight = None
        self.nll_loss = NLLLoss(weight=self.weight, ignore_index=self.padding_idx, reduction='mean')
        self.kl_loss = nn.KLDivLoss(reduction='mean')

        if self.use_gpu:
            self.cuda()
            self.weight = self.weight.cuda()

        self.fc1 = nn.Sequential(nn.Linear(self.hidden_size * 2, self.hidden_size), nn.Tanh())
        self.fc2 = nn.Sequential(nn.Linear(self.hidden_size * 2, self.hidden_size), nn.Tanh())

        # if self.with_bridge:
        #     self.fc1 = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size), nn.Tanh())
        #     self.fc2 = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size), nn.Tanh())
        #     self.fc3 = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size), nn.Tanh())
        # self.fc2 = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size), nn.Softmax(-1))

    def encode(self, inputs, hidden=None, is_training=False):
        """
        encode
        """
        outputs = Pack()
        src_inputs = _, lengths = inputs.src[0][:, 1:-1], inputs.src[1] - 2
        src_enc, (src_h, src_c) = self.encoder(src_inputs, hidden)
        # src_out = nn.AdaptiveAvgPool2d((1, src_enc.size(-1)))(src_enc)

        src_avg = nn.AdaptiveAvgPool2d((1, src_enc.size(-1)))(src_enc)
        src_max = nn.AdaptiveMaxPool2d((1, src_enc.size(-1)))(src_enc)
        src_out = self.fc1(torch.cat([src_avg, src_max], dim=-1))

        # if self.with_bridge:
        #     src_out = self.fc1(src_out)

        # knowledge
        batch_size, sent_num, sent = inputs.cue[0].size()
        cue_len = inputs.cue[1]
        cue_len[cue_len > 0] -= 2
        cue_inputs = inputs.cue[0].view(-1, sent)[:, 1:-1], cue_len.view(-1)
        cue_enc, (cue_h, cue_c) = self.kng_encoder(cue_inputs, hidden)
        # if self.with_bridge:
        #     cue_h=self.fc2(cue_h)
        cue_out = cue_h.view(batch_size, sent_num, -1)
        # cue_out = nn.AdaptiveAvgPool2d((1,cue_enc.size(-1)))(cue_enc).view(batch_size, sent_num, -1)

        # Attention
        src_cue, cue_attn = self.pri_attention(query=src_out,
                                               memory=cue_out,
                                               mask=inputs.cue[1].eq(0))
        cue_attn = cue_attn.squeeze(1)
        outputs.add(prior_attn=cue_attn)
        indexs = cue_attn.max(dim=1)[1]
        # hard attention
        if self.use_gs:
            knowledge = cue_out.gather(1, indexs.view(-1, 1, 1).repeat(1, 1, cue_out.size(-1)))
        else:
            knowledge = src_cue

        if self.use_posterior:
            tgt_inputs = inputs.tgt[0][:, 1:-1], inputs.tgt[1] - 2
            tgt_enc, (tgt_h, tgt_c) = self.tgt_encoder(tgt_inputs, hidden)
            # tgt_out = nn.AdaptiveAvgPool2d((1, tgt_enc.size(-1)))(tgt_enc)
            tgt_avg = nn.AdaptiveAvgPool2d((1, tgt_enc.size(-1)))(tgt_enc)
            tgt_max = nn.AdaptiveMaxPool2d((1, tgt_enc.size(-1)))(tgt_enc)
            tgt_out = self.fc1(torch.cat([tgt_avg, tgt_max], dim=-1))
            # if self.with_bridge:
            #     cue_out = self.fc3(cue_out)
            # P(z|u,r)
            # query=torch.cat([dec_init_hidden[-1], tgt_enc_hidden[-1]], dim=-1).unsqueeze(1)
            # P(z|r)
            tgt_cue, pos_attn = self.pos_attention(query=tgt_out,
                                                   memory=cue_out,
                                                   mask=inputs.cue[1].eq(0))
            pos_attn = pos_attn.squeeze(1)
            outputs.add(posterior_attn=pos_attn)

            if self.use_bow:
                bow_logits = self.bow_output_layer(knowledge)
                outputs.add(bow_logits=bow_logits)

        elif is_training:
            if self.use_gs:
                gumbel_attn = F.gumbel_softmax(torch.log(cue_attn + 1e-10), 0.1, hard=True)
                knowledge = torch.bmm(gumbel_attn.unsqueeze(1), cue_out)
                indexs = gumbel_attn.max(-1)[1]
            else:
                knowledge = src_cue

        outputs.add(indexs=indexs)
        if 'index' in inputs.keys():
            outputs.add(attn_index=inputs.index)

        if self.use_kd:
            knowledge = self.knowledge_dropout(knowledge)

        if self.weight_control:
            weights = (src_h[-1] * knowledge.squeeze(1)).sum(dim=-1)
            weights = self.sigmoid(weights)
            # norm in batch
            # weights = weights / weights.mean().item()
            outputs.add(weights=weights)
            knowledge = knowledge * weights.view(-1, 1, 1).repeat(1, 1, knowledge.size(-1))

        dec_init_state = self.decoder.initialize_state(
            hidden=(src_h, src_c),
            attn_memory=src_enc if self.attn_mode else None,
            memory_lengths=lengths if self.attn_mode else None,
            knowledge=knowledge)
        return outputs, dec_init_state

    def decode(self, input, state):
        """
        decode
        """
        log_prob, state, output = self.decoder.decode(input, state)
        return log_prob, state, output

    def forward(self, enc_inputs, dec_inputs, hidden=None, is_training=False):
        """
        forward
        """
        outputs, dec_init_state = self.encode(
            enc_inputs, hidden, is_training=is_training)
        log_probs, _ = self.decoder(dec_inputs, dec_init_state)
        outputs.add(logits=log_probs)
        return outputs

    def collect_metrics(self, outputs, target, epoch=-1):
        """
        collect_metrics
        """
        num_samples = target.size(0)
        metrics = Pack(num_samples=num_samples)
        loss = 0

        # test begin
        # nll = self.nll(torch.log(outputs.posterior_attn+1e-10), outputs.attn_index)
        # loss += nll
        # attn_acc = attn_accuracy(outputs.posterior_attn, outputs.attn_index)
        # metrics.add(attn_acc=attn_acc)
        # metrics.add(loss=loss)
        # return metrics
        # test end

        logits = outputs.logits
        scores = -self.nll_loss(logits, target, reduction=False)
        nll_loss = self.nll_loss(logits, target)
        num_words = target.ne(self.padding_idx).sum().item()
        acc = accuracy(logits, target, padding_idx=self.padding_idx)
        metrics.add(nll=(nll_loss, num_words), acc=acc)

        if self.use_posterior:
            kl_loss = self.kl_loss(torch.log(outputs.prior_attn + 1e-10),
                                   outputs.posterior_attn.detach())
            metrics.add(kl=kl_loss)
            if self.use_bow:
                bow_logits = outputs.bow_logits
                bow_labels = target[:, :-1]
                bow_logits = bow_logits.repeat(1, bow_labels.size(-1), 1)
                bow = self.nll_loss(bow_logits, bow_labels)
                loss += bow
                metrics.add(bow=bow)

            if epoch == -1 or epoch > self.pretrain_epoch or self.use_bow is not True:
                loss += nll_loss
                loss += kl_loss
        else:
            loss += nll_loss

        metrics.add(loss=loss)
        return metrics, scores

    def iterate(self, inputs, optimizer=None, grad_clip=None, is_training=False, epoch=-1):
        """
        iterate
        """
        enc_inputs = inputs
        dec_inputs = inputs.tgt[0][:, :-1], inputs.tgt[1] - 1
        target = inputs.tgt[0][:, 1:]

        outputs = self.forward(enc_inputs, dec_inputs, is_training=is_training)
        metrics, scores = self.collect_metrics(outputs, target, epoch=epoch)

        loss = metrics.loss
        if torch.isnan(loss):
            raise ValueError("nan loss encountered")

        if is_training:
            if self.use_pg:
                self.baseline = 0.99 * self.baseline + 0.01 * metrics.reward.item()
            assert optimizer is not None
            optimizer.zero_grad()
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                clip_grad_norm_(parameters=self.parameters(),
                                max_norm=grad_clip)
            optimizer.step()
        return metrics, scores

    def generate(self, batch_iter, num_batches=None):
        """
        generate
        """
        results = []
        batch_cnt = 0
        for batch in batch_iter:
            enc_outputs, preds, lengths, scores = self.forward(
                inputs=batch, enc_hidden=None)

            # denumericalization
            src = batch.src[0]
            tgt = batch.tgt[0]
            src = self.src_field.denumericalize(src)
            tgt = self.tgt_field.denumericalize(tgt)
            preds = self.tgt_field.denumericalize(preds)
            scores = scores.tolist()

            if 'cue' in batch:
                cue = self.tgt_field.denumericalize(batch.cue[0].data)
                enc_outputs.add(cue=cue)

            enc_outputs.add(src=src, tgt=tgt, preds=preds, scores=scores)
            result_batch = enc_outputs.flatten()
            results += result_batch
            batch_cnt += 1
            if batch_cnt == num_batches:
                break
        return results
