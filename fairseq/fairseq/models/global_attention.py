""" Global attention modules (Luong / Bahdanau) """
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .lstm import Linear
from ..modules.downsampled_multihead_attention import DownsampledMultiHeadAttention


# This class is mainly used by decoder.py for RNNs but also
# by the CNN / transformer decoder when copy attention is used
# CNN has its own attention mechanism ConvMultiStepAttention
# Transformer has its own MultiHeadedAttention


class ContextGeneratingAttentionLayer(nn.Module):
    def __init__(self, input_embed_dim, output_embed_dim):
        super().__init__()

        self.input_proj = Linear(input_embed_dim, output_embed_dim, bias=False)

    def forward(self, input, source_hids, source_hidinemb, encoder_padding_mask):
        # input: bsz x input_embed_dim
        # source_hids: srclen x bsz x output_embed_dim

        # x: bsz x output_embed_dim
        x = self.input_proj(input)

        # compute attention
        attn_scores = (source_hids * x.unsqueeze(0)).sum(dim=2)
        # don't attend over padding
        if encoder_padding_mask is not None:
            attn_scores = attn_scores.float().masked_fill_(
                encoder_padding_mask,
                #float('-inf')
                float('-9e9')  # L.P. FIX, DONE change this because when whole padding line tensors where nan
            ).type_as(attn_scores)  # FP16 support: cast to float and back
        attn_scores = F.softmax(attn_scores, dim=0)  # srclen x bsz

        # sum weighted sources
        x = (attn_scores.unsqueeze(2) * source_hidinemb).sum(dim=0)

        return x, attn_scores #just return the source context vector


class KeywordsAttention(nn.Module):
    """
    Global attention takes a matrix and a query vector. It
    then computes a parameterized convex combination of the matrix
    based on the input query.

    Constructs a unit mapping a query `q` of size `dim`
    and a source matrix `H` of size `n x dim`, to an output
    of size `dim`.


    .. mermaid::

       graph BT
          A[Query]
          subgraph RNN
            C[H 1]
            D[H 2]
            E[H N]
          end
          F[Attn]
          G[Output]
          A --> F
          C --> F
          D --> F
          E --> F
          C -.-> G
          D -.-> G
          E -.-> G
          F --> G

    All models compute the output as
    :math:`c = sum_{j=1}^{SeqLength} a_j H_j` where
    :math:`a_j` is the softmax of a score function.
    Then then apply a projection layer to [q, c].

    However they
    differ on how they compute the attention score.

    * Luong Attention (dot, general):
       * dot: :math:`score(H_j,q) = H_j^T q`
       * general: :math:`score(H_j, q) = H_j^T W_a q`


    * Bahdanau Attention (mlp):
       * :math:`score(H_j, q) = v_a^T tanh(W_a q + U_a h_j)`


    Args:
       dim (int): dimensionality of query and key
       coverage (bool): use coverage term
       attn_type (str): type of attention to use, options [dot,general,mlp]

    """

    def __init__(self, dim, attn_func="softmax"):
        super(KeywordsAttention, self).__init__()

        self.dim = dim
        assert attn_func in ["softmax", "sparsemax"], (
            "Please select a valid attention function.")
        self.attn_func = attn_func

        self.linear_condition = nn.Linear(dim, dim, bias=False)
        self.linear_context = nn.Linear(dim, dim, bias=False)
        self.linear_query = nn.Linear(dim, dim, bias=True)
        self.v = nn.Linear(dim, 1, bias=False)


    def score(self, h_t, h_s, h_cond):
        """
        Args:
          h_t (`FloatTensor`): sequence of queries `[batch x  dim]`
          h_s (`FloatTensor`): sequence of sources `[batch x src_len x dim]`
          h_cond (`FloatTensor`): input representation on which to condition `[batch x dim]`
        Returns:
          :obj:`FloatTensor`:
           raw attention scores (unnormalized) for each src index
          `[batch x tgt_len x src_len]`

        """

        # Check input sizes
        src_batch, src_len, src_dim = h_s.size()
        tgt_batch, tgt_dim = h_t.size()
        cond_batch, cond_dim = h_cond.size()
        assert src_batch==tgt_batch
        assert src_dim==tgt_dim
        assert self.dim==src_dim
        assert self.dim==src_dim

        dim = self.dim
        wq = self.linear_query(h_t)
        wq = wq.view(src_batch, 1, dim)
        wq = wq.expand(src_batch, src_len, dim)
        wq = wq.contiguous().view(-1, dim)

        uh = self.linear_context(h_s.contiguous().view(-1, dim))

        # however context needs to be expanded , do it here
        wcond = self.linear_condition(h_cond)
        wcond = wcond.view(src_batch, 1, dim)
        wcond = wcond.expand(src_batch, src_len, dim)
        wcond = wcond.contiguous().view(-1, dim)

        # (batch, t_len, s_len, d)
        wquh = torch.tanh(wq + uh + wcond)

        return self.v(wquh.view(-1, dim)).view(tgt_batch, src_len)


    def forward(self, source, memory_bank, condition, memory_lengths=None):
        """

        Args:
          source (`FloatTensor`): query vectors `[batch x tgt_len x dim]`
          memory_bank (`FloatTensor`): source vectors `[batch x src_len x dim]`
          memory_lengths (`LongTensor`): the source context lengths `[batch]`

        Returns:
          (`FloatTensor`, `FloatTensor`):

          * Computed vector `[tgt_len x batch x dim]`
          * Attention distribtutions for each query
             `[tgt_len x batch x src_len]`
        """

        # one step input
        if source.dim() == 2:
            one_step = True
        else:
            one_step = False

        batch, source_l, dim = memory_bank.size()
        batch_, dim_ = source.size()
        assert batch==batch_
        assert dim==dim_
        assert self.dim==dim

        # compute attention scores, as in Luong et al.
        align = self.score(source, memory_bank, condition)

        # Softmax or sparsemax to normalize attention weights
        if self.attn_func == "softmax":
            align_vectors = F.softmax(align.view(batch, source_l), -1)
        else:
            raise NotImplementedError
        align_vectors = align_vectors.view(batch, source_l).unsqueeze(1)

        # each context vector c_t is the weighted average
        # over all the source hidden states
        c = torch.bmm(align_vectors, memory_bank)
        c = c.squeeze()

        if one_step:
            assert batch==batch_

        return c, align_vectors # return the source context vector


class ContextAttention(nn.Module):

    def __init__(self, out_channels, embed_dim, num_heads, project_input=False, gated=False, downsample=False):
        super().__init__()
        self.attention = DownsampledMultiHeadAttention(
            out_channels, embed_dim, num_heads, dropout=0, bias=True,
            project_input=project_input, gated=gated, downsample=downsample,
        )
        self.in_proj_q = ContextLinear(out_channels, embed_dim)
        self.in_proj_k = ContextLinear(out_channels, embed_dim)
        self.in_proj_v = ContextLinear(out_channels, embed_dim)


    def forward(self, x, prevys, inmask, incremental_state):
        #residual = x
        query = self.in_proj_q(x)
        key = self.in_proj_k(prevys)
        value = self.in_proj_v(prevys)
        if incremental_state is not None:
            x, _ = self.attention(query, key, value, mask_future_timesteps=False,
                                  use_scalar_bias=True)  # , key_padding_mask=inmask)
        else:
            x, _ = self.attention(query, key, value, mask_future_timesteps=False, use_scalar_bias=True)#, key_padding_mask=inmask)
        return x

def ContextLinear(in_features, out_features, dropout=0.):
    """Weight-normalized Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features)
    m.weight.data.normal_(mean=0, std=math.sqrt((1 - dropout) / in_features))
    m.bias.data.zero_()
    return m