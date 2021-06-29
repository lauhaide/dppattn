""" Multi-Head Attention module """
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from onmt.utils.misc import generate_relative_positions_matrix,\
                            relative_matmul
# from onmt.utils.misc import aeq

from onmt.modules.multi_headed_attn import  MultiHeadedAttention

class MultiHeadedAttentionDPP(MultiHeadedAttention):

    def __init__(self, head_count, model_dim, dropout=0.1, max_relative_positions=0, dpp_rescaled=False):

        super(MultiHeadedAttentionDPP, self).__init__(head_count, model_dim,
                                                      dropout=dropout,
                                                      max_relative_positions=max_relative_positions)

        self.linear_past = nn.Linear(model_dim,
                                     head_count * self.dim_per_head)

        self._dpp_rescaled = dpp_rescaled


    def forward(self, key, value, query, past, mask=None,
                layer_cache=None, attn_type=None, dectemp=1.0):
        """
        Compute the context vector and the attention vectors.

        Args:
           key (FloatTensor): set of `key_len`
               key vectors ``(batch, key_len, dim)``
           value (FloatTensor): set of `key_len`
               value vectors ``(batch, key_len, dim)``
           query (FloatTensor): set of `query_len`
               query vectors  ``(batch, query_len, dim)``
           mask: binary mask 1/0 indicating which keys have
               zero / non-zero attention ``(batch, query_len, key_len)``
        Returns:
           (FloatTensor, FloatTensor):

           * output context vectors ``(batch, query_len, dim)``
           * one of the attention vectors ``(batch, query_len, key_len)``
        """

        # CHECKS
        # batch, k_len, d = key.size()
        # batch_, k_len_, d_ = value.size()
        # aeq(batch, batch_)
        # aeq(k_len, k_len_)
        # aeq(d, d_)
        # batch_, q_len, d_ = query.size()
        # aeq(batch, batch_)
        # aeq(d, d_)
        # aeq(self.model_dim % 8, 0)
        # if mask is not None:
        #    batch_, q_len_, k_len_ = mask.size()
        #    aeq(batch_, batch)
        #    aeq(k_len_, k_len)
        #    aeq(q_len_ == q_len)
        # END CHECKS

        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count
        key_len = key.size(1)
        query_len = query.size(1)

        def shape(x):
            """Projection."""
            return x.view(batch_size, -1, head_count, dim_per_head) \
                .transpose(1, 2)

        def unshape(x):
            """Compute context."""
            return x.transpose(1, 2).contiguous() \
                    .view(batch_size, -1, head_count * dim_per_head)

        # 1) Project key, value, and query.
        if layer_cache is not None:
            if attn_type == "self":
                query, key, value = self.linear_query(query),\
                                    self.linear_keys(query),\
                                    self.linear_values(query)
                key = shape(key)
                value = shape(value)
                if layer_cache["self_keys"] is not None:
                    key = torch.cat(
                        (layer_cache["self_keys"], key),
                        dim=2)
                if layer_cache["self_values"] is not None:
                    value = torch.cat(
                        (layer_cache["self_values"], value),
                        dim=2)
                layer_cache["self_keys"] = key
                layer_cache["self_values"] = value
            elif attn_type == "context":
                query = self.linear_query(query)
                if layer_cache["memory_keys"] is None:
                    key, value = self.linear_keys(key),\
                                 self.linear_values(value)
                    key = shape(key)
                    value = shape(value)
                else:
                    key, value = layer_cache["memory_keys"],\
                               layer_cache["memory_values"]
                layer_cache["memory_keys"] = key
                layer_cache["memory_values"] = value
        else:
            key = self.linear_keys(key)
            value = self.linear_values(value)
            query = self.linear_query(query)
            key = shape(key)
            value = shape(value)

        if self.max_relative_positions > 0 and attn_type == "self":
            key_len = key.size(2)
            # 1 or key_len x key_len
            relative_positions_matrix = generate_relative_positions_matrix(
                key_len, self.max_relative_positions,
                cache=True if layer_cache is not None else False)
            #  1 or key_len x key_len x dim_per_head
            relations_keys = self.relative_positions_embeddings(
                relative_positions_matrix.to(key.device))
            #  1 or key_len x key_len x dim_per_head
            relations_values = self.relative_positions_embeddings(
                relative_positions_matrix.to(key.device))

        query = shape(query)

        key_len = key.size(2)
        query_len = query.size(2)

        # DPP
        dpps = None
        if past is not None : # this class is used for context attn
            past = self.linear_past(past)
            past = shape(past)

            past_p2 = F.normalize(past, p=2, dim=3)
            key_p2 = F.normalize(key, p=2, dim=3)

            # Calculate past context similarity
            sim_prev = torch.matmul(past_p2, key_p2.transpose(2, 3))

            # compute dis-similarities
            dpps = 1 - (sim_prev ** 2)

        # 2) Calculate and scale scores.
        query = query / math.sqrt(dim_per_head)
        # batch x num_heads x query_len x key_len
        query_key = torch.matmul(query, key.transpose(2, 3))

        if self.max_relative_positions > 0 and attn_type == "self":
            scores = query_key + relative_matmul(query, relations_keys, True)
        else:
            scores = query_key
        scores = scores.float()

        if mask is not None:
            mask = mask.unsqueeze(1)  # [B, 1, 1, T_values]
            scores = scores.masked_fill(mask, -1e18)

        scores = scores / dectemp

        # 3) Apply attention dropout and compute context vectors.
        attn = self.softmax(scores).to(query.dtype)

        # past context diversity
        if dpps is not None:
            attn = attn * dpps
            if self._dpp_rescaled:
                attn = attn / attn.sum(3).unsqueeze(3)

        drop_attn = self.dropout(attn)

        context_original = torch.matmul(drop_attn, value)

        context = unshape(context_original)

        new_past = unshape(torch.matmul(drop_attn, key)) # LP) can choose previous context based on keys

        output = self.final_linear(context)
        # CHECK
        # batch_, q_len_, d_ = output.size()
        # aeq(q_len, q_len_)
        # aeq(batch, batch_)
        # aeq(d, d_)

        ## Return one attn
        top_attn = attn \
            .view(batch_size, head_count,
                  query_len, key_len)[:, 0, :, :] \
            .contiguous()

        return output, top_attn, new_past

class MultiHeadedAttentionDPPPrevst(MultiHeadedAttention):

    def __init__(self, head_count, model_dim, dropout=0.1, max_relative_positions=0, dpp_rescaled=False):

        super(MultiHeadedAttentionDPPPrevst, self).__init__(head_count, model_dim,
                                                      dropout=dropout,
                                                      max_relative_positions=max_relative_positions)

        self.linear_past = nn.Linear(model_dim,
                                     head_count * self.dim_per_head)

        self._dpp_rescaled = dpp_rescaled

    def forward(self, key, value, query, past, mask=None,
                layer_cache=None, attn_type=None, dectemp=1.0):
        """
        Compute the context vector and the attention vectors.

        Args:
           key (FloatTensor): set of `key_len`
               key vectors ``(batch, key_len, dim)``
           value (FloatTensor): set of `key_len`
               value vectors ``(batch, key_len, dim)``
           query (FloatTensor): set of `query_len`
               query vectors  ``(batch, query_len, dim)``
           mask: binary mask 1/0 indicating which keys have
               zero / non-zero attention ``(batch, query_len, key_len)``
        Returns:
           (FloatTensor, FloatTensor):

           * output context vectors ``(batch, query_len, dim)``
           * one of the attention vectors ``(batch, query_len, key_len)``
        """

        # CHECKS
        # batch, k_len, d = key.size()
        # batch_, k_len_, d_ = value.size()
        # aeq(batch, batch_)
        # aeq(k_len, k_len_)
        # aeq(d, d_)
        # batch_, q_len, d_ = query.size()
        # aeq(batch, batch_)
        # aeq(d, d_)
        # aeq(self.model_dim % 8, 0)
        # if mask is not None:
        #    batch_, q_len_, k_len_ = mask.size()
        #    aeq(batch_, batch)
        #    aeq(k_len_, k_len)
        #    aeq(q_len_ == q_len)
        # END CHECKS

        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count
        key_len = key.size(1)
        query_len = query.size(1)

        def shape(x):
            """Projection."""
            return x.view(batch_size, -1, head_count, dim_per_head) \
                .transpose(1, 2)

        def unshape(x):
            """Compute context."""
            return x.transpose(1, 2).contiguous() \
                    .view(batch_size, -1, head_count * dim_per_head)

        # 1) Project key, value, and query.
        if layer_cache is not None:
            if attn_type == "self":
                query, key, value = self.linear_query(query),\
                                    self.linear_keys(query),\
                                    self.linear_values(query)
                key = shape(key)
                value = shape(value)
                if layer_cache["self_keys"] is not None:
                    key = torch.cat(
                        (layer_cache["self_keys"], key),
                        dim=2)
                if layer_cache["self_values"] is not None:
                    value = torch.cat(
                        (layer_cache["self_values"], value),
                        dim=2)
                layer_cache["self_keys"] = key
                layer_cache["self_values"] = value
            elif attn_type == "context":
                query = self.linear_query(query)
                if layer_cache["memory_keys"] is None:
                    key, value = self.linear_keys(key),\
                                 self.linear_values(value)
                    key = shape(key)
                    value = shape(value)
                else:
                    key, value = layer_cache["memory_keys"],\
                               layer_cache["memory_values"]
                layer_cache["memory_keys"] = key
                layer_cache["memory_values"] = value
        else:
            key = self.linear_keys(key)
            value = self.linear_values(value)
            query = self.linear_query(query)
            key = shape(key)
            value = shape(value)

        if self.max_relative_positions > 0 and attn_type == "self":
            key_len = key.size(2)
            # 1 or key_len x key_len
            relative_positions_matrix = generate_relative_positions_matrix(
                key_len, self.max_relative_positions,
                cache=True if layer_cache is not None else False)
            #  1 or key_len x key_len x dim_per_head
            relations_keys = self.relative_positions_embeddings(
                relative_positions_matrix.to(key.device))
            #  1 or key_len x key_len x dim_per_head
            relations_values = self.relative_positions_embeddings(
                relative_positions_matrix.to(key.device))

        query = shape(query)

        key_len = key.size(2)
        query_len = query.size(2)

        # DPP
        dpps = None
        if past is not None : # this class is used for context attn
            past = self.linear_past(past)
            past = shape(past)

            past_p2 = F.normalize(past, p=2, dim=3)
            key_p2 = F.normalize(key, p=2, dim=3)

            sim_prev = torch.matmul(past_p2, key_p2.transpose(2, 3))

            # compute dis-similarities
            dpps = 1 - (sim_prev ** 2)

        # 2) Calculate and scale scores.
        query = query / math.sqrt(dim_per_head)
        # batch x num_heads x query_len x key_len
        query_key = torch.matmul(query, key.transpose(2, 3))

        if self.max_relative_positions > 0 and attn_type == "self":
            scores = query_key + relative_matmul(query, relations_keys, True)
        else:
            scores = query_key
        scores = scores.float()

        if mask is not None:
            mask = mask.unsqueeze(1)  # [B, 1, 1, T_values]
            scores = scores.masked_fill(mask, -1e18)

        scores = scores / dectemp

        # 3) Apply attention dropout and compute context vectors.
        attn = self.softmax(scores).to(query.dtype)

        # past context diversity
        if dpps is not None:
            attn = attn * dpps
            if self._dpp_rescaled:
                attn = attn / attn.sum(3).unsqueeze(3)

        drop_attn = self.dropout(attn)

        context_original = torch.matmul(drop_attn, value)

        context = unshape(context_original)

        output = self.final_linear(context)
        # CHECK
        # batch_, q_len_, d_ = output.size()
        # aeq(q_len, q_len_)
        # aeq(batch, batch_)
        # aeq(d, d_)

        # Return one attn
        top_attn = attn \
            .view(batch_size, head_count,
                  query_len, key_len)[:, 0, :, :] \
            .contiguous()

        return output, top_attn, None

class MultiHeadedAttentionDPP_P(MultiHeadedAttention):
    """ This class and MultiHeadedAttentionDPP differ in that *past* context vector uses the same projection as *keys*"""

    def __init__(self, head_count, model_dim, dropout=0.1, max_relative_positions=0, dpp_rescaled=False):

        super(MultiHeadedAttentionDPP_P, self).__init__(head_count, model_dim,
                                                      dropout=dropout,
                                                      max_relative_positions=max_relative_positions)

        self._dpp_rescaled = dpp_rescaled


    def forward(self, key, value, query, past, mask=None,
                layer_cache=None, attn_type=None, dectemp=1.0):
        """
        Compute the context vector and the attention vectors.

        Args:
           key (FloatTensor): set of `key_len`
               key vectors ``(batch, key_len, dim)``
           value (FloatTensor): set of `key_len`
               value vectors ``(batch, key_len, dim)``
           query (FloatTensor): set of `query_len`
               query vectors  ``(batch, query_len, dim)``
           mask: binary mask 1/0 indicating which keys have
               zero / non-zero attention ``(batch, query_len, key_len)``
        Returns:
           (FloatTensor, FloatTensor):

           * output context vectors ``(batch, query_len, dim)``
           * one of the attention vectors ``(batch, query_len, key_len)``
        """

        # CHECKS
        # batch, k_len, d = key.size()
        # batch_, k_len_, d_ = value.size()
        # aeq(batch, batch_)
        # aeq(k_len, k_len_)
        # aeq(d, d_)
        # batch_, q_len, d_ = query.size()
        # aeq(batch, batch_)
        # aeq(d, d_)
        # aeq(self.model_dim % 8, 0)
        # if mask is not None:
        #    batch_, q_len_, k_len_ = mask.size()
        #    aeq(batch_, batch)
        #    aeq(k_len_, k_len)
        #    aeq(q_len_ == q_len)
        # END CHECKS

        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count
        key_len = key.size(1)
        query_len = query.size(1)

        def shape(x):
            """Projection."""
            return x.view(batch_size, -1, head_count, dim_per_head) \
                .transpose(1, 2)

        def unshape(x):
            """Compute context."""
            return x.transpose(1, 2).contiguous() \
                    .view(batch_size, -1, head_count * dim_per_head)

        # 1) Project key, value, and query.
        if layer_cache is not None:
            if attn_type == "self":
                query, key, value = self.linear_query(query),\
                                    self.linear_keys(query),\
                                    self.linear_values(query)
                key = shape(key)
                value = shape(value)
                if layer_cache["self_keys"] is not None:
                    key = torch.cat(
                        (layer_cache["self_keys"], key),
                        dim=2)
                if layer_cache["self_values"] is not None:
                    value = torch.cat(
                        (layer_cache["self_values"], value),
                        dim=2)
                layer_cache["self_keys"] = key
                layer_cache["self_values"] = value
            elif attn_type == "context":
                query = self.linear_query(query)
                if layer_cache["memory_keys"] is None:
                    key, value = self.linear_keys(key),\
                                 self.linear_values(value)
                    key = shape(key)
                    value = shape(value)
                else:
                    key, value = layer_cache["memory_keys"],\
                               layer_cache["memory_values"]
                layer_cache["memory_keys"] = key
                layer_cache["memory_values"] = value
        else:
            key = self.linear_keys(key)
            value = self.linear_values(value)
            query = self.linear_query(query)
            key = shape(key)
            value = shape(value)

        if self.max_relative_positions > 0 and attn_type == "self":
            key_len = key.size(2)
            # 1 or key_len x key_len
            relative_positions_matrix = generate_relative_positions_matrix(
                key_len, self.max_relative_positions,
                cache=True if layer_cache is not None else False)
            #  1 or key_len x key_len x dim_per_head
            relations_keys = self.relative_positions_embeddings(
                relative_positions_matrix.to(key.device))
            #  1 or key_len x key_len x dim_per_head
            relations_values = self.relative_positions_embeddings(
                relative_positions_matrix.to(key.device))

        query = shape(query)

        key_len = key.size(2)
        query_len = query.size(2)

        # DPP
        dpps = None
        if past is not None : # this class is used for context attn
            past = self.linear_keys(past) # use same projection as for keys
            past = shape(past)

            past_p2 = F.normalize(past, p=2, dim=3)
            key_p2 = F.normalize(key, p=2, dim=3)

            sim_prev = torch.matmul(past_p2, key_p2.transpose(2, 3))

            # compute dis-similarities
            dpps = 1 - (sim_prev ** 2)

        # 2) Calculate and scale scores.
        query = query / math.sqrt(dim_per_head)
        # batch x num_heads x query_len x key_len
        query_key = torch.matmul(query, key.transpose(2, 3))

        if self.max_relative_positions > 0 and attn_type == "self":
            scores = query_key + relative_matmul(query, relations_keys, True)
        else:
            scores = query_key
        scores = scores.float()

        if mask is not None:
            mask = mask.unsqueeze(1)  # [B, 1, 1, T_values]
            scores = scores.masked_fill(mask, -1e18)

        scores = scores / dectemp

        # 3) Apply attention dropout and compute context vectors.
        attn = self.softmax(scores).to(query.dtype)

        # past context diversity
        if dpps is not None:
            attn = attn * dpps
            if self._dpp_rescaled:
                attn = attn / attn.sum(3).unsqueeze(3)

        drop_attn = self.dropout(attn)

        context_original = torch.matmul(drop_attn, value)

        context = unshape(context_original)

        new_past = unshape(torch.matmul(drop_attn, key)) # LP) can choose previous context based on keys

        output = self.final_linear(context)
        # CHECK
        # batch_, q_len_, d_ = output.size()
        # aeq(q_len, q_len_)
        # aeq(batch, batch_)
        # aeq(d, d_)

        ## Return one attn
        top_attn = attn \
            .view(batch_size, head_count,
                  query_len, key_len)[:, 0, :, :] \
            .contiguous()

        return output, top_attn, new_past

class MultiHeadedAttentionDPPExpQK(MultiHeadedAttention):
    """ This class is newer implementation of the kernel, not fully tested but no better results... """

    def __init__(self, head_count, model_dim, dropout=0.1, max_relative_positions=0, dpp_rescaled=False):

        super(MultiHeadedAttentionDPPExpQK, self).__init__(head_count, model_dim,
                                                      dropout=dropout,
                                                      max_relative_positions=max_relative_positions)

        self.linear_past = nn.Linear(model_dim,
                                     head_count * self.dim_per_head)

        self._dpp_rescaled = dpp_rescaled

    def forward(self, key, value, query, past, mask=None,
                layer_cache=None, attn_type=None, dectemp=1.0):
        """
        Compute the context vector and the attention vectors.

        Args:
           key (FloatTensor): set of `key_len`
               key vectors ``(batch, key_len, dim)``
           value (FloatTensor): set of `key_len`
               value vectors ``(batch, key_len, dim)``
           query (FloatTensor): set of `query_len`
               query vectors  ``(batch, query_len, dim)``
           mask: binary mask 1/0 indicating which keys have
               zero / non-zero attention ``(batch, query_len, key_len)``
        Returns:
           (FloatTensor, FloatTensor):

           * output context vectors ``(batch, query_len, dim)``
           * one of the attention vectors ``(batch, query_len, key_len)``
        """

        # CHECKS
        # batch, k_len, d = key.size()
        # batch_, k_len_, d_ = value.size()
        # aeq(batch, batch_)
        # aeq(k_len, k_len_)
        # aeq(d, d_)
        # batch_, q_len, d_ = query.size()
        # aeq(batch, batch_)
        # aeq(d, d_)
        # aeq(self.model_dim % 8, 0)
        # if mask is not None:
        #    batch_, q_len_, k_len_ = mask.size()
        #    aeq(batch_, batch)
        #    aeq(k_len_, k_len)
        #    aeq(q_len_ == q_len)
        # END CHECKS

        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count
        key_len = key.size(1)
        query_len = query.size(1)

        def shape(x):
            """Projection."""
            return x.view(batch_size, -1, head_count, dim_per_head) \
                .transpose(1, 2)

        def unshape(x):
            """Compute context."""
            return x.transpose(1, 2).contiguous() \
                    .view(batch_size, -1, head_count * dim_per_head)



        # 1) Project key, value, and query.
        if layer_cache is not None:
            if attn_type == "self":
                query, key, value = self.linear_query(query),\
                                    self.linear_keys(query),\
                                    self.linear_values(query)
                key = shape(key)
                value = shape(value)
                if layer_cache["self_keys"] is not None:
                    key = torch.cat(
                        (layer_cache["self_keys"], key),
                        dim=2)
                if layer_cache["self_values"] is not None:
                    value = torch.cat(
                        (layer_cache["self_values"], value),
                        dim=2)
                layer_cache["self_keys"] = key
                layer_cache["self_values"] = value
            elif attn_type == "context":
                query = self.linear_query(query)
                if layer_cache["memory_keys"] is None:
                    key, value = self.linear_keys(key),\
                                 self.linear_values(value)
                    key = shape(key)
                    value = shape(value)
                else:
                    key, value = layer_cache["memory_keys"],\
                               layer_cache["memory_values"]
                layer_cache["memory_keys"] = key
                layer_cache["memory_values"] = value
        else:
            key = self.linear_keys(key)
            value = self.linear_values(value)
            query = self.linear_query(query)
            key = shape(key)
            value = shape(value)

        if self.max_relative_positions > 0 and attn_type == "self":
            key_len = key.size(2)
            # 1 or key_len x key_len
            relative_positions_matrix = generate_relative_positions_matrix(
                key_len, self.max_relative_positions,
                cache=True if layer_cache is not None else False)
            #  1 or key_len x key_len x dim_per_head
            relations_keys = self.relative_positions_embeddings(
                relative_positions_matrix.to(key.device))
            #  1 or key_len x key_len x dim_per_head
            relations_values = self.relative_positions_embeddings(
                relative_positions_matrix.to(key.device))

        query = shape(query)

        key_len = key.size(2)
        query_len = query.size(2)

        # DPP
        dpps = None
        if past is not None : # this class is used for context attn
            past = self.linear_past(past)
            past = shape(past)

            #past_p2 = F.normalize(past, p=2, dim=3)
            #key_p2 = F.normalize(key, p=2, dim=3)
            #sim_prev = torch.matmul(past_p2, key_p2.transpose(2, 3))

            #kernels
            #https://mycourses.aalto.fi/pluginfile.php/537774/mod_resource/content/3/CS-E4830_Lecture1.pdf

            #pp = torch.cdist(past.contiguous().view(batch_size*head_count, -1, dim_per_head),
            #                 key.contiguous().view(batch_size*head_count, -1, dim_per_head), p=2)

            #https://cmry.github.io/notes/euclidean-v-cosine
            #https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065

            # A) memory
            #differences = past.unsqueeze(3) - key.unsqueeze(2)
            #distances = torch.sum(differences **2, -1)

            # B) Quadratic expansion of ||u-v||^2
            # https://math.stackexchange.com/questions/391796/how-to-expand-equation-inside-the-l2-norm
            past_p2 = torch.norm(past, p=2, dim=3).view(batch_size, head_count, -1, 1)
            key_p2 = torch.norm(key, p=2, dim=3).view(batch_size, head_count, 1, -1)
            #print('past_p2',past_p2.shape )
            #print('key_p2',key_p2.shape)

            distances = past_p2 + key_p2 - 2.0 * torch.matmul(past, torch.transpose(key, 2,3))
            distances = torch.clamp(distances, 0.0, float('inf'))

            K = torch.exp(-0.5 * distances) # [b x ts x d] and [b x ss x d]

            K = K.view(batch_size, head_count, query_len, key_len)

            # compute dis-similarities L'+eps*I with eps=0.01 for positive definiteness
            # https://arxiv.org/pdf/1511.05077.pdf
            dpps =  1.0 - (K ** 2)  #1.0201
            dpps = torch.clamp(dpps, 0.00001, float('inf'))

            if query_len>1: # i.e. it's trainig time            
              mask_init_step = torch.zeros(*dpps.size()).to(past.device)
              mask_init_step[:,:,0,:].fill_(1)
              dpps = dpps.masked_fill(mask_init_step.bool(), 1.0)



        # 2) Calculate and scale scores.
        query = query / math.sqrt(dim_per_head)
        # batch x num_heads x query_len x key_len
        query_key = torch.matmul(query, key.transpose(2, 3))

        if self.max_relative_positions > 0 and attn_type == "self":
            scores = query_key + relative_matmul(query, relations_keys, True)
        else:
            scores = query_key
        scores = scores.float()

        if mask is not None:
            mask = mask.unsqueeze(1)  # [B, 1, 1, T_values]
            scores = scores.masked_fill(mask, -1e18)

        scores = scores / dectemp

        # 3) Apply attention dropout and compute context vectors.
        attn = self.softmax(scores).to(query.dtype)

        # past context diversity
        if dpps is not None:
            attn = attn * dpps
            if self._dpp_rescaled:
                attn = attn / attn.sum(3).unsqueeze(3)

        drop_attn = self.dropout(attn)

        context_original = torch.matmul(drop_attn, value)

        context = unshape(context_original)

        new_past = unshape(torch.matmul(drop_attn, key)) # LP) can choose previous context based on keys

        output = self.final_linear(context)
        # CHECK
        # batch_, q_len_, d_ = output.size()
        # aeq(q_len, q_len_)
        # aeq(batch, batch_)
        # aeq(d, d_)

        ## Return one attn
        top_attn = attn \
            .view(batch_size, head_count,
                  query_len, key_len)[:, 0, :, :] \
            .contiguous()

        return output, top_attn, new_past
