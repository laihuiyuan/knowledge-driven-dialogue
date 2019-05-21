# -*- coding: UTF-8 -*-

import torch
import numpy as np
import torch.nn as nn

from source.modules.embedder import Embedder
from source.modules.transformer.sub_layers import EncoderLayer
from source.modules.transformer.sub_layers import DecoderLayer


def no_pad_mask(seq, padding_idx):
    assert seq.dim() == 2
    return seq.ne(padding_idx).type(torch.float).unsqueeze(-1)

def attn_key_pad_mask(seq_q, seq_k, padding_idx):
    ''' 
    For masking out the padding part of key sequence. 
    '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(padding_idx)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask


def subsequent_mask(seq):
    ''' 
    For masking out the subsequent info. 
    '''

    bat, len = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len, len), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(bat, -1, -1)  # b x ls x ls

    return subsequent_mask

class PosEmbedder(nn.Module):
    def __init__(self, max_seq_len, d_word_vec):
        super(PosEmbedder, self).__init__()
        pos_enc = np.array(
            [[pos / np.power(10000, 2.0 * (j // 2) / d_word_vec) for j in range(d_word_vec)]
            for pos in range(max_seq_len)])
        pos_enc[:, 0::2] = np.sin(pos_enc[:, 0::2])
        pos_enc[:, 1::2] = np.cos(pos_enc[:, 1::2])
        pad_row = np.zeros([1, d_word_vec])
        pos_enc = np.concatenate([pad_row, pos_enc]).astype(np.float32)

        # additional single row for PAD idx
        self.pos_enc = nn.Embedding(max_seq_len + 1, d_word_vec)
        # fix positional encoding: exclude weight from grad computation
        self.pos_enc.weight = nn.Parameter(torch.from_numpy(pos_enc), requires_grad=False)

    def forward(self, input_len):
        max_len = max(input_len)
        tensor = torch.cuda.LongTensor if input_len.is_cuda else torch.LongTensor
        input_pos = tensor([list(range(1, len+1)) + [0]*(max_len-len).item() for len in input_len])

        return self.pos_enc(input_pos)


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(self, embedder, pos_embedder, d_k, d_v, d_model,
                 d_inner, n_layers, n_head, padding_idx, dropout=0.3):
        super().__init__()

        self.embedder = embedder
        self.pos_embedder = pos_embedder
        self.padding_idx = padding_idx
        # self.n_position = max_seq_len + 1

        # self.position_enc = nn.Embedding.from_pretrained(
        #     position_embedding(self.n_position, word_vec_dim, padding_idx=0),freeze=True)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, inputs, src_pos, return_attn=False):

        enc_slf_attn_list = []

        # Prepare masks
        non_pad_mask = no_pad_mask(inputs,self.padding_idx)
        slf_att_mask = attn_key_pad_mask(seq_q=inputs, seq_k=inputs, padding_idx=self.padding_idx)

        # Forward
        enc_output = self.embedder(inputs) + self.pos_embedder(src_pos)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_att_mask)
            if return_attn:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attn:
            return enc_output, enc_slf_attn_list
        return enc_output


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(self, tgt_vocab_size, max_seq_len, word_vec_dim, dec_embedder, pos_embedder,
                 n_layers, n_head, d_k, d_v, d_model, d_inner, padding_idx=0, dropout=0.3):
        super().__init__()

        self.tgt_vocab_size = tgt_vocab_size
        self.max_seq_len = max_seq_len
        self.dec_embedder = dec_embedder
        self.pos_embedder = pos_embedder
        self.word_vec_dim = word_vec_dim
        self.padding_idx = padding_idx
        # self.n_position = max_seq_len + 1

        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, inputs, tgt_pos, enc_inputs, enc_output, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # Prepare masks
        non_pad_mask = no_pad_mask(inputs,self.padding_idx)

        slf_attn_mask_subseq = subsequent_mask(inputs)
        slf_attn_mask_keypad = attn_key_pad_mask(inputs, inputs, self.padding_idx)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        dec_enc_attn_mask = attn_key_pad_mask(seq_q=inputs, seq_k=enc_inputs, padding_idx=self.padding_idx)

        # Forward
        dec_output = self.dec_embedder(inputs) + self.pos_embedder(tgt_pos)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_mask)

            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]
                dec_enc_attn_list += [dec_enc_attn]

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,


class BaseModel(nn.Module):
    """
    BaseModel
    """
    def __init__(self):
        super(BaseModel, self).__init__()

    def forward(self, *input):
        """
        forward
        """
        raise NotImplementedError

    def __repr__(self):
        main_string = super(BaseModel, self).__repr__()
        num_parameters = sum([p.nelement() for p in self.parameters()])
        main_string += "\nNumber of parameters: {}\n".format(num_parameters)
        return main_string

    def save(self, filename):
        """
        save
        """
        torch.save(self.state_dict(), filename)
        print("Saved model state to '{}'!".format(filename))

    def load(self, filename):
        """
        load
        """
        if os.path.isfile(filename):
            state_dict = torch.load(
                filename, map_location=lambda storage, loc: storage)
            self.load_state_dict(state_dict, strict=False)
            print("Loaded model state from '{}'".format(filename))
        else:
            print("Invalid model state file: '{}'".format(filename))

class Transformer(BaseModel):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self,
            src_vocab_size, tgt_vocab_size, max_seq_len,word_vec_dim=300,d_model=300, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, padding_idx=0, dropout=0.1,tie_embedding=True,
            tgt_emb_prj_weight_sharing=True,emb_src_tgt_weight_sharing=True):
        super(Transformer, self).__init__()

        self.src_vocab_size=src_vocab_size
        self.tgt_vocab_size=tgt_vocab_size
        self.max_seq_len=max_seq_len
        self.word_vec_dim=word_vec_dim
        self.n_position = max_seq_len + 1
        self.padding_idx=padding_idx
        self.tie_embedding=tie_embedding

        enc_embedder = Embedder(num_embeddings=self.src_vocab_size,
                                embedding_dim=self.embed_size,
                                padding_idx=self.padding_idx)

        if self.tie_embedding:
            assert self.src_vocab_size == self.tgt_vocab_size
            dec_embedder = enc_embedder
        else:
            dec_embedder = Embedder(num_embeddings=self.src_vocab_size,
                                    embedding_dim=self.embed_size,
                                    padding_idx=self.padding_idx)


        pos_embedder = nn.Embedding.from_pretrained(
            position_embedding(self.n_position, word_vec_dim, padding_idx=0),freeze=True)

        self.encoder = Encoder(
            src_vocab_size=src_vocab_size, max_seq_len=max_seq_len,word_vec_dim=word_vec_dim,
            enc_embedder=enc_embedder,pos_embedder=pos_embedder,d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,padding_idx=padding_idx,dropout=dropout)

        self.decoder = Decoder(
            tgt_vocab_size=tgt_vocab_size, max_seq_len=max_seq_len, word_vec_dim=word_vec_dim,
            dec_embedder=dec_embedder, pos_embedder=pos_embedder, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v, padding_idx=padding_idx, dropout=dropout)

        self.tgt_word_prj = nn.Linear(d_model, tgt_vocab_size, bias=False)
        nn.init.xavier_normal_(self.tgt_word_prj.weight)

        assert d_model == word_vec_dim, \
            'To facilitate the residual connections, \
             the dimensions of all module output shall be the same.'

        if tgt_emb_prj_weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.tgt_word_prj.weight = self.decoder.tgt_word_emb.weight
            self.x_logit_scale = (d_model ** -0.5)
        else:
            self.x_logit_scale = 1.

        if emb_src_tgt_weight_sharing:
            # Share the weight matrix between source & target word embeddings
            assert src_vocab_size == tgt_vocab_size, \
                "To share word embedding table, the vocabulary size of src/tgt shall be the same."
            self.encoder.src_word_emb.weight = self.decoder.tgt_word_emb.weight

    def forward(self, src_seq, src_pos, tgt_seq, tgt_pos):

        tgt_seq, tgt_pos = tgt_seq[:, :-1], tgt_pos[:, :-1]

        enc_output, *_ = self.encoder(src_seq, src_pos)
        dec_output, *_ = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output)
        seq_logit = self.tgt_word_prj(dec_output) * self.x_logit_scale

        return seq_logit.view(-1, seq_logit.size(2))

    def collect_metrics(self, outputs, target, epoch=-1):
        """
        collect_metrics
        """
        num_samples = target.size(0)
        metrics = Pack(num_samples=num_samples)
        loss = 0

        # test begin
        # nll = self.nll(torch.log(output.posterior_attn+1e-10), output.attn_index)
        # loss += nll
        # attn_acc = attn_accuracy(output.posterior_attn, output.attn_index)
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