import torch.nn as nn
from .MultiHeadedAttention import MultiHeadedAttention
from .PositionwiseFeedForward import PositionwiseFeedForward
from .PositionalEncoding import PositionalEncoding
from .EncoderDecoder import EncoderDecoder
from .Encoder import Encoder
from .EncoderLayer import EncoderLayer
from .Decoder import Decoder
from .DecoderLayer import DecoderLayer
from .Embeddings import Embeddings
from .Generator import Generator

from .common import copy
from .logger import add_logging

def make_model(
    src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1, log_on = False
):
    "#SOL(辅助函数): 从超参数构建模型"
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N), # HL: encoder
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N), # HL: decoder
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)), # HL: source embeddings
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)), # HL: target embeddings
        Generator(d_model, tgt_vocab), # HL: generator
    )

    # See https://arxiv.org/pdf/1502.01852.pdf
    # NT: 初始化网络权重 (Glorot / fan_avg)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p) # [-limit, limit] (limit = \sqrt(fan_in + fan_out))

    # ✅ 如果需要 logging，给每个子模块加 log wrapper
    if log_on:
        for name, module in model.named_modules():
            add_logging(module, enabled=True)
    return model




""""
EncoderDecoder(
  (encoder): Encoder(
    (layers): ModuleList(
      (0-1): 2 x EncoderLayer(
        (self_attn): MultiHeadedAttention(
          (linears): ModuleList(
            (0-3): 4 x Linear(in_features=512, out_features=512, bias=True)
          )
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (sublayer): ModuleList(
          (0-1): 2 x SublayerConnection(
            (norm): LayerNorm()
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
    )
    (norm): LayerNorm()
  )
  (decoder): Decoder(
    (layers): ModuleList(
      (0-1): 2 x DecoderLayer(
        (self_attn): MultiHeadedAttention(
          (linears): ModuleList(
            (0-3): 4 x Linear(in_features=512, out_features=512, bias=True)
          )
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (src_attn): MultiHeadedAttention(
          (linears): ModuleList(
            (0-3): 4 x Linear(in_features=512, out_features=512, bias=True)
          )
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (sublayer): ModuleList(
          (0-2): 3 x SublayerConnection(
            (norm): LayerNorm()
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
    )
    (norm): LayerNorm()
  )
  (src_embed): Sequential(
    (0): Embeddings(
      (lut): Embedding(11, 512)
    )
    (1): PositionalEncoding(
      (dropout): Dropout(p=0.1, inplace=False)
    )
  )
  (tgt_embed): Sequential(
    (0): Embeddings(
      (lut): Embedding(11, 512)
    )
    (1): PositionalEncoding(
      (dropout): Dropout(p=0.1, inplace=False)
    )
  )
  (generator): Generator(
    (proj): Linear(in_features=512, out_features=11, bias=True)
  )
)
"""