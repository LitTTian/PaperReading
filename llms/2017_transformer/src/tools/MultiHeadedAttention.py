import torch.nn as nn
from .common import clones, attention
from .logger import log

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    # @log
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        # seq_len = query.size(1)
        # print(nbatches, seq_len, self.h, self.d_k)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            # BUGFIX: x.size(1)保留seq_len维度
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            # lin(x).view(nbatches, seq_len, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]
        # print(query.shape, key.shape, value.shape) # torch.Size([16, 8, 128, 96])
        # print(mask.shape if mask is not None else "No mask")

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )
        # print("x shape after attention:", x.shape) # torch.Size([16, 16, 8, 128, 96])

        # 3) "Concat" using a view and apply a final linear.
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
            # .view(nbatches, seq_len, self.h * self.d_k)
        )
        del query
        del key
        del value
        return self.linears[-1](x)