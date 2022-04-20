import torch
import torch.nn as nn
import math
from torch.autograd import Variable
from torch.nn import functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        scale = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * scale)
        pe[:, 1::2] = torch.cos(position * scale)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=True)
        return self.dropout(x)


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab, maxlen):
        super(Embeddings, self).__init__()
        self.token_embedding = nn.Embedding(vocab, d_model)
        self.pos_embedding = nn.Embedding(maxlen, d_model)
        self.d_model = d_model
        self.maxlen = maxlen

    def forward(self, x):
        positions = self.pos_embedding(torch.arange(start=0, end=self.maxlen))
        tokens = self.token_embedding(x)
        return (positions + tokens) * math.sqrt(self.d_model)


class PositionalWiseFFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionalWiseFFN, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return F.relu(self.w_2(F.relu(self.w_1(x))))


def ScaledDotProduct(query, key, values, dropout=None):
    d_k = query.size(-1)
    scores = query @ key.transpose(-2, -1) / math.sqrt(d_k)
    p_atten = F.softmax(scores, dim=-1)
    if dropout:
        p_atten = dropout(p_atten)
    return p_atten @ values, p_atten


class MultiheadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiheadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.heads = list()
        self.attn = None
        for _ in range(h+1):
            self.heads.append(nn.Linear(d_model, d_model))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value):
        batch_size = query.size(0)
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.heads, (query, key, value))]
        x, self.attn = ScaledDotProduct(query, key, value, self.dropout)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        return self.heads[-1](x)


class LayerNorm(nn.Module):
    "Construct a layernorm module"
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class sublayerConnectionAttention(nn.Module):
    def __init__(self, h, d_model, dropout_head=0.1, dropout_connection=0.1):
        super(sublayerConnectionAttention, self).__init__()
        self.multiheads = MultiheadAttention(h, d_model, dropout_head)
        self.layernorm = LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout_connection)

    def forward(self, query, key, value):
        original = torch.cat([query, key, value], dim=-1)
        x = self.multiheads(query, key, value)
        x = self.dropout(self.layernorm(x))
        return x + original


class sublayerConnectionFFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout_ffn=0.1, dropout_connection=0.1):
        super(sublayerConnectionFFN, self).__init__()
        self.ffn = PositionalWiseFFN(d_model, d_ff, dropout_ffn)
        self.layernorm = LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout_connection)

    def forward(self, x):
        original = x
        x = self.dropout(self.ffn(x))
        return x + original


class Classifier(nn.Module):
    def __init__(self, d_model, n_class):
        super(Classifier, self).__init__()
        self.classifier = nn.Linear(d_model, n_class)

    def forward(self, x):
        return self.classifier(x)


class Model(nn.Module):
    def __init__(self,
                 n_class,
                 vocab,
                 n_layers=6,
                 h=2,
                 d_model=512,
                 d_ff=128,
                 maxlen=200,
                 dropout_encodings=0.1,
                 dropout_connection_attention=0.1,
                 dropout_connection_ffn=0.1,
                 dropout_attention=0.1,
                 dropout_ffn=0.1):
        super(Model, self).__init__()
        self.input_embeddings = Embeddings(d_model, vocab, maxlen)
        self.input_encodings = PositionalEncoding(d_model, dropout_encodings, maxlen)
        self.sublayer_attention = list()
        self.sublayer_ffn = list()
        for _ in range(n_layers):
            self.sublayer_attention.append(sublayerConnectionAttention(
                h, d_model, dropout_attention, dropout_connection_attention))
            self.sublayer_ffn.append(sublayerConnectionFFN(
                d_model, d_ff, dropout_ffn, dropout_connection_ffn))
        self.classifier = Classifier(d_model, n_class)
        self.n_layers = n_layers

    def forward(self, x):
        embeddings = self.input_embeddings(x)
        encodings = self.input_encodings(x)
        x = embeddings + encodings
        for i in range(self.n_layers):
            x = self.sublayer_attention[i](x, x, x)
            x = self.sublayer_ffn[i](x)
        outputs = self.classifier(x)
        return outputs

