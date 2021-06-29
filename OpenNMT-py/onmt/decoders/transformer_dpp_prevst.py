"""
Implementation of "Attention is All You Need"
"""

import torch
import torch.nn as nn

from onmt.decoders.decoder import DecoderBase
from onmt.modules import MultiHeadedAttention, AverageAttention, MultiHeadedAttentionDPPPrevst
from onmt.modules.position_ffn import PositionwiseFeedForward
from onmt.utils.misc import sequence_mask

from onmt.decoders.transformer import TransformerDecoder

class TransformerDecoderLayerDPPPrevst(nn.Module):
    """
    Args:
      d_model (int): the dimension of keys/values/queries in
          :class:`MultiHeadedAttention`, also the input size of
          the first-layer of the :class:`PositionwiseFeedForward`.
      heads (int): the number of heads for MultiHeadedAttention.
      d_ff (int): the second-layer of the :class:`PositionwiseFeedForward`.
      dropout (float): dropout probability.
      self_attn_type (string): type of self-attention scaled-dot, average
    """

    def __init__(self, d_model, heads, d_ff, dropout, attention_dropout,
                 self_attn_type="scaled-dot", max_relative_positions=0,
                 aan_useffn=False, dpp_rescaled=False, single_dpp_head=False):
        super(TransformerDecoderLayerDPPPrevst, self).__init__()

        print("*    DPP attention with dpp_rescaled={} and 1head={}".format(dpp_rescaled, single_dpp_head))

        if self_attn_type == "scaled-dot":
            self.self_attn = MultiHeadedAttention(
                heads, d_model, dropout=dropout,
                max_relative_positions=max_relative_positions)
        elif self_attn_type == "average":
            self.self_attn = AverageAttention(d_model,
                                              dropout=attention_dropout,
                                              aan_useffn=aan_useffn)

        self.context_attn = MultiHeadedAttentionDPPPrevst(
            1 if single_dpp_head else heads,
            d_model, dropout=attention_dropout, dpp_rescaled=dpp_rescaled)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)
        self.drop = nn.Dropout(dropout)

    def forward(self, inputs, memory_bank, src_pad_mask, tgt_pad_mask, prev_context,
                layer_cache=None, step=None, dectemp=1.0):
        """
        Args:
            inputs (FloatTensor): ``(batch_size, 1, model_dim)``
            memory_bank (FloatTensor): ``(batch_size, src_len, model_dim)``
            src_pad_mask (LongTensor): ``(batch_size, 1, src_len)``
            tgt_pad_mask (LongTensor): ``(batch_size, 1, 1)``

        Returns:
            (FloatTensor, FloatTensor):

            * output ``(batch_size, 1, model_dim)``
            * attn ``(batch_size, 1, src_len)``

        """
        dec_mask = None
        if step is None:
            tgt_len = tgt_pad_mask.size(-1)
            future_mask = torch.ones(
                [tgt_len, tgt_len],
                device=tgt_pad_mask.device,
                dtype=torch.uint8)
            future_mask = future_mask.triu_(1).view(1, tgt_len, tgt_len)
            # BoolTensor was introduced in pytorch 1.2
            try:
                future_mask = future_mask.bool()
            except AttributeError:
                pass
            dec_mask = torch.gt(tgt_pad_mask + future_mask, 0)

        input_norm = self.layer_norm_1(inputs)

        if isinstance(self.self_attn, MultiHeadedAttention):
            query, attn = self.self_attn(input_norm, input_norm, input_norm,
                                         mask=dec_mask,
                                         layer_cache=layer_cache,
                                         attn_type="self")
        elif isinstance(self.self_attn, AverageAttention):
            query, attn = self.self_attn(input_norm, mask=dec_mask,
                                         layer_cache=layer_cache, step=step)

        query = self.drop(query) + inputs

        query_norm = self.layer_norm_2(query)
        mid, attn, prev_context = self.context_attn(memory_bank, memory_bank, query_norm, prev_context,
                                      mask=src_pad_mask,
                                      layer_cache=layer_cache,
                                      attn_type="context",
                                      dectemp=dectemp)
        output = self.feed_forward(self.drop(mid) + query)

        return output, attn, prev_context

    def update_dropout(self, dropout, attention_dropout):
        self.self_attn.update_dropout(attention_dropout)
        self.context_attn.update_dropout(attention_dropout)
        self.feed_forward.update_dropout(dropout)
        self.drop.p = dropout


class TransformerDecoderDPPPrevst(TransformerDecoder):

    def __init__(self, num_layers, d_model, heads, d_ff,
                 copy_attn, self_attn_type, dropout, attention_dropout,
                 embeddings, max_relative_positions, aan_useffn, coverage_attn=None,
                 dpp_rescaled=False, single_dpp_head=False):
        super(TransformerDecoderDPPPrevst, self).__init__(num_layers, d_model, heads, d_ff,
                 copy_attn, self_attn_type, dropout, attention_dropout,
                 embeddings, max_relative_positions, aan_useffn, coverage_attn=coverage_attn)

        self.embeddings = embeddings

        # Decoder State
        self.state = {}

        self.transformer_layers = nn.ModuleList(
            [TransformerDecoderLayerDPPPrevst(d_model, heads, d_ff, dropout,
             attention_dropout, copy_attn, self_attn_type=self_attn_type,
             max_relative_positions=max_relative_positions,
             aan_useffn=aan_useffn,
             dpp_rescaled=dpp_rescaled,
             single_dpp_head=single_dpp_head)
             for i in range(num_layers)])

        # previously, there was a GlobalAttention module here for copy
        # attention. But it was never actually used -- the "copy" attention
        # just reuses the context attention.
        self._copy = copy_attn
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        # LP) add coverage loss
        self._coverage = coverage_attn

    def forward(self, tgt, memory_bank, step=None, prev_key_contexts=None, dectemp=1.0, **kwargs):
        """Decode, possibly stepwise."""
        if step == 0:
            self._init_cache(memory_bank)

        tgt_words = tgt[:, :, 0].transpose(0, 1)

        emb = self.embeddings(tgt, step=step)
        assert emb.dim() == 3  # len x batch x embedding_dim

        output = emb.transpose(0, 1).contiguous()
        src_memory_bank = memory_bank.transpose(0, 1).contiguous()

        pad_idx = self.embeddings.word_padding_idx
        src_lens = kwargs["memory_lengths"]
        src_max_len = self.state["src"].shape[0]
        src_pad_mask = ~sequence_mask(src_lens, src_max_len).unsqueeze(1)
        tgt_pad_mask = tgt_words.data.eq(pad_idx).unsqueeze(1)  # [B, 1, T_tgt]

        for i, layer in enumerate(self.transformer_layers):
            layer_cache = self.state["cache"]["layer_{}".format(i)] \
                if step is not None else None

            # dpp context
            prev_x = None
            if i > 0 and step is None: #training
                # key_context has is of size [b x t x d], need to shift time steps to be previous, then accumulate up-to
                dstidx = torch.arange(1, output.size()[1]).long().cuda()
                oriidx = torch.arange(output.size()[1] - 1).long().cuda()
                prev_x = output.new(*output.size()).fill_(0)
                prev_x[:, dstidx] = torch.index_select(output, 1, oriidx)  # [b x ts x d]
                prev_x = torch.cumsum(prev_x, 1)  # debugged, sums on correct dim per batch
                prev_x[:, 0, :] = 1
                step_x = output.new(output.size()[0], output.size()[1], 1).fill_(0)
                step_x[:, 1:] = dstidx.unsqueeze(1)
                step_x[:, 0, :] = 1
                prev_x = prev_x / step_x  # averaged sum

            elif i > 0 and step is not None and step > 0:  # test after second time step
                prev_x = prev_key_contexts[i - 1] / step  # recover context from previous layer [b x 1 x d], debugged

            elif i > 0 and step is not None and step == 0:  # test first time step
                prev_x = src_memory_bank.new(*output.size()).fill_(1)

            output, attn, key_contexts = layer(
                output,
                src_memory_bank,
                src_pad_mask,
                tgt_pad_mask,
                prev_x,
                layer_cache=layer_cache,
                step=step,
                dectemp=dectemp)

            # dpp context
            if step is not None:
                if step>0: #test after second step
                    prev_key_contexts[i] += output
                else:
                    if prev_key_contexts is None:
                        prev_key_contexts = []
                    prev_key_contexts.append(output)

        output = self.layer_norm(output)
        dec_outs = output.transpose(0, 1).contiguous()
        cove_attn = None
        attn = attn.transpose(0, 1).contiguous()

        attns = {"std": attn}
        if self._copy:
            attns["copy"] = attn
        if self._coverage:
            print("*    never here for dpp transformer!")
            exit()
            attns["coverage"] = cove_attn

        # TODO change the way attns is returned dict => list or tuple (onnx)
        if step is None:
            return dec_outs, attns # on training do not need to keep the context from one step to the other
        else:
            return dec_outs, attns, prev_key_contexts

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.dec_layers,
            opt.dec_rnn_size,
            opt.heads,
            opt.transformer_ff,
            opt.copy_attn,
            opt.self_attn_type,
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            opt.attention_dropout[0] if type(opt.attention_dropout)
            is list else opt.dropout,
            embeddings,
            opt.max_relative_positions,
            opt.aan_useffn,
            opt.lambda_coverage>0.0, # LP) if this is greater than 0, this means the coverage loss will be used
            opt.dpp_rescaled,
            opt.single_dpp_head)