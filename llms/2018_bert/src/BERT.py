import torch
from torch import nn
import math

# MODULE: common Blocks
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

# === Embeddings ===
class JointEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_position=512, type_vocab_size=2):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, token_type_ids=None):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

# === Multi-Head Attention (与 HuggingFace 对齐) ===
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        q = self.transpose_for_scores(self.query(hidden_states))
        k = self.transpose_for_scores(self.key(hidden_states))
        v = self.transpose_for_scores(self.value(hidden_states))

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            scores = scores + attention_mask  # mask 已经是 -inf 或 0

        attn_probs = nn.Softmax(dim=-1)(scores)
        attn_probs = self.dropout(attn_probs)

        context = torch.matmul(attn_probs, v)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(hidden_states.size(0), -1, self.num_heads * self.head_dim)

        return context

# === Feed Forward ===
class FeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dropout=0.1):
        super().__init__()
        self.dense1 = nn.Linear(hidden_size, intermediate_size)
        self.act = nn.GELU()
        self.dense2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.dense2(self.act(self.dense1(x))))

# === Encoder Layer ===
class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size, dropout=0.1):
        super().__init__()
        self.attention = nn.Module()
        self.attention.self = MultiHeadAttention(hidden_size, num_heads, dropout)
        self.attention.output = nn.Module()
        self.attention.output.dense = nn.Linear(hidden_size, hidden_size)
        self.attention.output.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        # MODULE: feed forward
        self.intermediate = nn.Module()
        self.intermediate.dense = nn.Linear(hidden_size, intermediate_size)
        self.output = nn.Module()
        self.output.dense = nn.Linear(intermediate_size, hidden_size)
        self.output.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Attention
        attn_output = self.attention.self(x, mask)
        attn_output = self.attention.output.dense(attn_output)
        x = self.attention.output.LayerNorm(x + attn_output)

        # Feed Forward
        ffn_output = self.intermediate.dense(x)
        ffn_output = nn.GELU()(ffn_output)
        ffn_output = self.output.dense(ffn_output)
        ffn_output = self.dropout(ffn_output)
        x = self.output.LayerNorm(x + ffn_output)
        return x

# === Encoder Stack ===
class Encoder(nn.Module):
    def __init__(self, num_layers, hidden_size, num_heads, intermediate_size, dropout=0.1):
        super().__init__()
        self.layer = nn.ModuleList([
            EncoderLayer(hidden_size, num_heads, intermediate_size, dropout) for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        for layer in self.layer:
            x = layer(x, mask)
        return x


class BertPredictionHead(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        # MLM部分
        self.predictions = nn.Module()
        self.predictions.transform = nn.Module()
        self.predictions.transform.dense = nn.Linear(hidden_size, hidden_size)
        self.predictions.transform.activation = nn.GELU()
        self.predictions.transform.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.predictions.decoder = nn.Linear(hidden_size, vocab_size, bias=False)
        self.predictions.bias = nn.Parameter(torch.zeros(vocab_size))

        # NSP部分
        self.seq_relationship = nn.Linear(hidden_size, 2)

    def forward(self, encoder_output, pooled_output, mlm=False, nsp=False):
        prediction = None
        seq_pred = None

        if mlm:
            x = self.predictions.transform.dense(encoder_output)
            x = self.predictions.transform.activation(x)
            x = self.predictions.transform.LayerNorm(x)
            x = self.predictions.decoder(x)
            prediction = x + self.predictions.bias

        if nsp:
            seq_pred = self.seq_relationship(pooled_output)

        return prediction, seq_pred
# === BERT ===
from collections import namedtuple
BertOutput = namedtuple("BertOutput", ["last_hidden_state", "pooler_output", "mlm_logits", "seq_relationship"])
class BERT(nn.Module):
    def __init__(self, vocab_size, hidden_size=768, num_heads=12, num_layers=12, intermediate_size=3072):
        super().__init__()
        self.bert = nn.Module() # HL: 为了和官方命名对齐
        self.bert.embeddings = JointEmbedding(vocab_size, hidden_size)
        self.bert.encoder = Encoder(num_layers, hidden_size, num_heads, intermediate_size)

        # Pooler
        # self.pooler = nn.Linear(hidden_size, hidden_size)
        # self.pooler_act = nn.Tanh()
        self.bert.pooler = nn.Module()
        self.bert.pooler.dense = nn.Linear(hidden_size, hidden_size)
        self.bert.pooler.activation = nn.Tanh()

        # MLM, NSP
        # self.cls = nn.Module()
        # self.cls.predictions = nn.Module()
        # self.cls.predictions.transform = nn.Module()
        # self.cls.predictions.transform.dense = nn.Linear(hidden_size, hidden_size)
        # self.cls.predictions.transform.activation = nn.GELU()
        # self.cls.predictions.transform.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        # self.cls.predictions.decoder = nn.Linear(hidden_size, vocab_size, bias=False)
        # self.cls.predictions.bias = nn.Parameter(torch.zeros(vocab_size))
        # self.cls.seq_relationship = nn.Linear(hidden_size, 2)
        self.cls = BertPredictionHead(hidden_size, vocab_size)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, mlm=False, nsp=False):
        embedding_output = self.bert.embeddings(input_ids, token_type_ids)
        encoder_output = self.bert.encoder(embedding_output, attention_mask)
        pooled_output = self.bert.pooler.activation(self.bert.pooler.dense(encoder_output[:, 0]))
        # outputs = (encoder_output, pooled_output)
        if mlm or nsp:
            mlm_logits, nsp_logits = self.cls(encoder_output, pooled_output, mlm, nsp)
        # if mlm:
        #     prediction = self.cls.predictions.transform(encoder_output)
        #     prediction = self.cls.predictions.decoder(prediction)
        #     prediction = prediction + self.cls.predictions.bias
        #     outputs = outputs + (prediction,)

        # if nsp:
        #     seq_pred = self.cls.seq_relationship(pooled_output)
        #     outputs = outputs + (seq_pred,)
        # outputs += cls_outputs
        return BertOutput(
            last_hidden_state=encoder_output,
            pooler_output=pooled_output,
            mlm_logits=mlm_logits if mlm else None,
            seq_relationship=nsp_logits if nsp else None
        )
