import torch
import torch.nn as nn
import copy
import math

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    """
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # NT: scores [batch_size, num_heads, seq_len, seq_len]
    # NT: 在Transformer Decoder中，由于右边的上下文的mask设置为0，所以我们说Transformer Decoder是单向的（只能关注到左边的上下文）
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
