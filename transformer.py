"""Base class for encoders and generic multi encoders."""
# This codes is borrowed from OpenNMT.


import math
import torch
import torch.nn as nn
import pdb


class PositionwiseFeedForward(nn.Module):
    """ A two-layer Feed-Forward-Network with residual layer norm.

        Args:
            d_model (int): the size of input for the first-layer of the FFN.
            d_ff (int): the hidden layer size of the second-layer
                              of the FNN.
            dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Layer definition.

        Args:
            input: [ batch_size, input_len, model_dim ]

        Returns:
            output: [ batch_size, input_len, model_dim ]
        """
        hidden = self.relu(self.w_1(x))
        inter  = self.dropout(hidden)
        output = self.w_2(inter)
        
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=5000):
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(1)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        #self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None, no_emb=False):
        emb = emb * math.sqrt(self.dim)
        #pdb.set_trace()
        if step is None:
            if no_emb:
                emb = self.pe[:emb.size(0)]
            else:
                emb = emb + self.pe[:emb.size(0)]
        else:
            if no_emb:
                emb = self.pe[step]
            else:
                emb = emb + self.pe[step]
        #emb = self.dropout(emb)

        return emb

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=1024, dropout=0.0,
                 learned_pos=True, padding_idx=0, sparse=False):

        super(LearnedPositionalEncoding, self).__init__()
        self.pe = nn.Embedding(max_len, dim, padding_idx=padding_idx, sparse=sparse)
        self.padding_idx = padding_idx
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def make_positions(self, tensor, onnx_trace: bool = False):
        """Replace non-padding symbols with their position numbers.

        Position numbers begin at padding_idx+1. Padding symbols are ignored.
        """
        # The series of casts and type-conversions here are carefully
        # balanced to both work with ONNX export and XLA. In particular XLA
        # prefers ints, cumsum defaults to output longs, and ONNX doesn't know
        # how to handle the dtype kwarg in cumsum.
        mask = tensor.ne(self.padding_idx).int()
        return (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + self.padding_idx

    def forward(self, emb, step=None, no_emb=False):
        # 采用学习的位置编码，实质上位置编码是可学习的参数。
        pos = self.make_positions(emb.mean(-1))
        if step is None:
            if no_emb:
                emb = self.pe(pos)
            else:
                emb = emb + self.pe(pos)
        else:
            if no_emb:
                emb = self.pe(pos[step])
            else:
                emb = emb + self.pe(pos[step])
        emb = self.dropout(emb)

        return emb

class Embeddings(nn.Module):
    def __init__(self, word_vec_size,
                       word_vocab_size,
                       word_padding_idx,
                       position_encoding=False,
                       dropout=0,
                       sparse=False):

        self.word_padding_idx = word_padding_idx

        self.word_vec_size = word_vec_size
        
        embedding = nn.Embedding(word_vocab_size, word_vec_size,\
                                 padding_idx=word_padding_idx, sparse=sparse)
        
        self.embedding_size = word_vec_size
        
        super(Embeddings, self).__init__()
        self.make_embedding = nn.Sequential()
        self.make_embedding.add_module('word', embedding)

        self.position_encoding = position_encoding

        if self.position_encoding:
            pe = PositionalEncoding(dropout, self.embedding_size)
            self.make_embedding.add_module('pe', pe)

    @property
    def word_lut(self):
        """ word look-up table """
        return self.make_embedding[0]

    def load_pretrained_vectors(self, emb_file, fixed):
        """Load in pretrained embeddings.

        Args:
            emb_file (str) : path to torch serialized embeddings
            fixed (bool) : if true, embeddings are not updated
        """
        if emb_file:
            pretrained = torch.load(emb_file)
            pretrained_vec_size = pretrained.size(1)
            if self.word_vec_size > pretrained_vec_size:
                self.word_lut.weight.data[:, :pretrained_vec_size] = pretrained
            elif self.word_vec_size < pretrained_vec_size:
                self.word_lut.weight.data \
                        .copy_(pretrained[:, :self.word_vec_size])
            else:
                self.word_lut.weight.data.copy_(pretrained)
            if fixed:
                self.word_lut.weight.requires_grad = False

    def forward(self, source, step=None,
                pe_only=False, we_only=False):
        """
        Computes the embeddings for words and features.

        Args:
                source (`LongTensor`): index tensor `[len x batch]`
        Return:
                `FloatTensor`: word embeddings `[len x batch x embedding_size]`
        """
        if pe_only:
            return self.make_embedding[1](source, step=step, no_emb=True)
        
        if we_only:
            return self.make_embedding[0](source, step=step)

        if self.position_encoding:
            for i, module in enumerate(self.make_embedding._modules.values()):
                if i == len(self.make_embedding._modules.values()) - 1:
                    source = module(source, step=step)
                else:
                    source = module(source)
        else:
            source = self.make_embedding(source)

        return source


class MultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention module from
    "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`.

    Similar to standard `dot` attention but uses
    multiple attention distributions simulataneously
    to select relevant items.

    Args:
       head_count (int): number of parallel heads
       model_dim (int): the dimension of keys/values/queries,
           must be divisible by head_count
       dropout (float): dropout parameter
    """

    def __init__(self, head_count, model_dim, dropout=0.1):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim    = model_dim

        super(MultiHeadedAttention, self).__init__()
        self.head_count = head_count

        self.linear_keys   = nn.Linear(model_dim,
                                       head_count * self.dim_per_head)
        self.linear_values = nn.Linear(model_dim,
                                       head_count * self.dim_per_head)
        self.linear_query  = nn.Linear(model_dim,
                                       head_count * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(model_dim, model_dim)

    def forward(self, key, value, query, mask=None,
                layer_cache=None, type=None):
        """
        Compute the context vector and the attention vectors.

        Args:
           key (`FloatTensor`): set of `key_len`
                key vectors `[batch, key_len, dim]`
           value (`FloatTensor`): set of `key_len`
                value vectors `[batch, key_len, dim]`
           query (`FloatTensor`): set of `query_len`
                 query vectors  `[batch, query_len, dim]`
           mask: binary mask indicating which keys have
                 non-zero attention `[batch, query_len, key_len]`
        Returns:
           (`FloatTensor`, `FloatTensor`) :

           * output context vectors `[batch, query_len, dim]`
           * one of the attention vectors `[batch, query_len, key_len]`
        """

        batch_size   = key.size(0)
        dim_per_head = self.dim_per_head
        head_count   = self.head_count
        key_len      = key.size(1)
        query_len    = query.size(1)

        def shape(x):
            """  projection """
            return x.view(batch_size, -1, head_count, dim_per_head) \
                         .transpose(1, 2)

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous() \
                    .view(batch_size, -1, head_count * dim_per_head)

        # 1) Project key, value, and query.
        if layer_cache is not None:
            if type == "self":
                query, key, value = self.linear_query(query), \
                                    self.linear_keys(key),    \
                                    self.linear_values(value)

                key   = shape(key)
                value = shape(value)

                if layer_cache is not None:
                    device = key.device
                    if layer_cache["self_keys"] is not None:
                        key = torch.cat(
                               (layer_cache["self_keys"].to(device), key),
                                dim=2)
                    if layer_cache["self_values"] is not None:
                        value = torch.cat(
                            (layer_cache["self_values"].to(device), value),
                             dim=2)
                    layer_cache["self_keys"]   = key
                    layer_cache["self_values"] = value
            elif type == "context":
                query = self.linear_query(query)
                if layer_cache is not None:
                    if layer_cache["memory_keys"] is None:
                        key, value = self.linear_keys(key),\
                                     self.linear_values(value)
                        key   = shape(key)
                        value = shape(value)
                    else:
                        key, value = layer_cache["memory_keys"],\
                                     layer_cache["memory_values"]
                    layer_cache["memory_keys"]   = key
                    layer_cache["memory_values"] = value
                else:
                    key, value = self.linear_keys(key),\
                                 self.linear_values(value)
                    key   = shape(key)
                    value = shape(value)
        else:
            key   = self.linear_keys(key)
            value = self.linear_values(value)
            query = self.linear_query(query)
            key   = shape(key)
            value = shape(value)

        query     = shape(query)

        key_len   = key.size(2)
        query_len = query.size(2)

        # 2) Calculate and scale scores.
        query  = query / math.sqrt(dim_per_head)
        scores = torch.matmul(query, key.transpose(2, 3))

        if mask is not None:
            mask   = mask.unsqueeze(1)  # [B, 1, 1, T_values]
            scores = scores.masked_fill(mask, -1e18)

        # 3) Apply attention dropout and compute context vectors.
        attn      = self.softmax(scores)
        drop_attn = self.dropout(attn)
        context   = unshape(torch.matmul(drop_attn, value))

        output    = self.final_linear(context)
        # CHECK
        # batch_, q_len_, d_ = output.size()
        # aeq(q_len, q_len_)
        # aeq(batch, batch_)
        # aeq(d, d_)

        # Return one attn
        top_attn = attn \
            .view(batch_size, head_count,
                  query_len, key_len)[:, 0, :, :].contiguous()

        return output, top_attn


class SelfAttentionLayer(nn.Module):
    """
      原始的Transformer的self-attention层。
    """

    def __init__(self, d_model, heads, d_ff, dropout):
        super(SelfAttentionLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
                         heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)
        self.drop_1 = nn.Dropout(dropout)
        self.drop_2 = nn.Dropout(dropout)

    def forward(self, inputs, mask=None, mask_rnn=None):

        inputs_ln  = self.layer_norm_1(inputs)
        
        context, _ = self.self_attn(inputs_ln, inputs_ln, 
                                    inputs_ln, mask=mask)

        out    = context + inputs
        out_ln = self.layer_norm_2(out)

        output = self.feed_forward(out_ln)
        output = self.drop_2(output) + out
        
        return output

class TransformerEncoder(nn.Module):
    """
    Args:
        num_layers (int): number of encoder layers
        d_model (int): size of the model
        heads (int): number of heads
        d_ff (int): size of the inner FF layer
        dropout (float): dropout parameters
        embeddings (:obj:`onmt.modules.Embeddings`):
          embeddings to use, should have positional encodings

    Returns:
        (`FloatTensor`, `FloatTensor`):

        * embeddings `[src_len x batch_size x model_dim]`
        * memory_bank `[src_len x batch_size x model_dim]`
    """

    def __init__(self, num_layers=3, d_model=128,
                       heads=1, d_ff=256,
                       dropout=0.1, embeddings=None,
                       layer_type='self'):
        super(TransformerEncoder, self).__init__()

        self.num_layers  = num_layers
        self.layer_norm  = nn.LayerNorm(d_model, eps=1e-6)
        self.pe = PositionalEncoding(dim=d_model)
        #self.pe = LearnedPositionalEncoding(dim=d_model)
        #self or self-gru
        self.layer_type = layer_type

        if self.layer_type == 'self':
            #
            self.transformer = nn.ModuleList([SelfAttentionLayer(d_model,
                                     int(d_model/16), d_model*4, dropout)
                                       for _ in range(self.num_layers)])

    def forward(self, src, mask=None, use_pe=True):
        """ See :obj:`EncoderBase.forward()`"""
        # src.shape = [sentence_length, batch_size]
        # use_pe 
        emb = src
        output = emb.transpose(0, 1).contiguous()
        if use_pe:
            pe_emb = self.pe(output)
            output = output + pe_emb
        words  = src[:, :, 0].transpose(0, 1)
        w_batch, w_len = words.size()
        
        #pdb.set_trace()
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # [B, 1, T]   
        # Run the forward pass of every layer of the tranformer.
        for i in range(self.num_layers):
            output = self.transformer[i](output, mask)
        
        output = self.layer_norm(output)

        return output.transpose(0, 1).contiguous()

