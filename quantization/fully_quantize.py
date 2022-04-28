# quantization: weight & activations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from collections import defaultdict


class LayerNormQuant(nn.Module):
    "Construct a layernorm module"

    def __init__(self, features, eps=1e-6):
        super(LayerNormQuant, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features, device="cuda"))
        self.b_2 = nn.Parameter(torch.zeros(features, device="cuda"))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        numer_quant = apply_quantization_activations("", (x - mean), False)
        denominator_quant = apply_quantization_activations("", (std + self.eps), False)
        return self.a_2 * numer_quant / denominator_quant + self.b_2

# assuming h=8 and num_heads=2 -> we have 77 activations pending for quantization!
buckets = defaultdict(lambda: [None, None])

keys = {"scale_product_head_0",
        "scale_product_head_0_q",
        "scale_product_head_0_k",
        "scale_product_head_0_v",
        "scale_product_head_1",
        "scale_product_head_1_q",
        "scale_product_head_1_k",
        "scale_product_head_1_v",
        "mha0_v",
        "mha1_v",
        "mha0_k",
        "mha1_k",
        "mha0_q",
        "mha1_q",
        "mha0",
        "mha1",
        "ffn0_input",
        "ffn1_input",
        "ffn0_output",
        "ffn1_output",
        "classifier"}


def quantization_weights(X, k=8):
    xmax = torch.max(X).item()
    xmin = torch.min(X).item()
    s = (xmax - xmin) / (2 ** k - 1)
    q = torch.div(torch.clamp(X, min=xmin, max=xmax), s, rounding_mode="floor") * s + xmin
    return q


class Quantization_Weights(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, k=8):
        return quantization_weights(inputs, k)

    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs, None, None


_apply_quantization_weights = Quantization_Weights.apply

"""
Original representation
"""


def quantization_activations(X, xmin, xmax, k=8, EMA=True):
    if xmin is None and xmax is None:
        xmin = torch.min(X).item()
        xmax = torch.max(X).item()
    if X.requires_grad and EMA:
        if len(X.size()) > 3:
            num_heads = X.size(1)
            x_copy = X.permute(0, 2, 3, 1).contiguous().view(-1, num_heads)
            xmin = 0.9 * xmin + 0.1 * \
                   torch.min(x_copy, dim=0)[0].detach().unsqueeze(-1).unsqueeze(-1).expand(X.size(1), X.size(2), X.size(3))
            xmax = 0.9 * xmax + 0.1 * \
                   torch.max(x_copy, dim=0)[0].detach().unsqueeze(-1).unsqueeze(-1).expand(X.size(1), X.size(2), X.size(3))
        else:
            xmin = 0.9 * xmin + 0.1 * torch.min(X).item()
            xmax = 0.9 * xmax + 0.1 * torch.max(X).item()
    s = (xmax - xmin) / (2 ** k - 1)
    q = torch.div(torch.clamp(X, min=xmin, max=xmax), s, rounding_mode="trunc") * s + xmin
    return q, xmin, xmax


"""
Does activations have backpropagation?
"""
class Quantization_Activations(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, xmin, xmax, k=8, EMA=True):
        return quantization_activations(X, xmin, xmax, k, EMA)

    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs, None, None


_quantization_activations = Quantization_Activations.apply


def apply_quantization_activations(key_name, X, EMA=True):
    if key_name in keys:
        xmin, xmax = buckets[key_name]
    else:
        xmin, xmax = None, None
    if EMA and X.requires_grad:
        assert key_name in keys
        xmin, xmax = buckets[key_name]
        if xmin is None and xmax is None:
            if len(X.size()) > 3:
                # It may happen at the multi-head attention
                num_heads = X.size(1)
                x_copy = X.permute(0, 2, 3, 1).contiguous().view(-1, num_heads)
                xmin = torch.min(x_copy, dim=0)[0].detach().unsqueeze(-1).unsqueeze(-1)
                xmax = torch.max(x_copy, dim=0)[0].detach().unsqueeze(-1).unsqueeze(-1)
            else:
                xmin = torch.min(X).item()
                xmax = torch.max(X).item()

            buckets[key_name] = [xmin, xmax]
    X, xmin_updated, xmax_updated = _quantization_activations(X, xmin, xmax, EMA)
    buckets[key_name] = [xmin_updated, xmax_updated]
    return X


class PositionalWiseFFNQuant(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1, head=0):
        super(PositionalWiseFFNQuant, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.head = head

    def forward(self, x):
        key_input = "ffn{}".format(self.head) + "_input"
        key_output = "ffn{}".format(self.head) + "_output"
        x = apply_quantization_activations(key_input, x)

        x = self.dropout(F.relu(self.w_1(x)))
        x = apply_quantization_activations(key_output, x)
        return self.w_2(x)


def ScaledDotProductQuant(query, key, values, dropout=None, mask=None, head=0):
    common = "scale_product_head_{}".format(head)
    query_key = common + "_q"
    key_key = common + "_k"
    value_key = common + "_v"
    attention_key = common
    query = apply_quantization_activations(query_key, query)
    key = apply_quantization_activations(key_key, key)
    values = apply_quantization_activations(value_key, values)

    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        mask = mask.squeeze(1)
        scores = scores.masked_fill_(mask == 0, -1e-9)
    p_atten = F.softmax(scores, dim=-1)
    if dropout:
        p_atten = dropout(p_atten)
    p_atten = apply_quantization_activations(attention_key, p_atten)
    return torch.matmul(p_atten, values), p_atten


class MultiheadAttentionQuant(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, head=0):
        super(MultiheadAttentionQuant, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.heads = list()
        self.attn = None
        for _ in range(h):
            self.heads.append(nn.Linear(d_model, d_model).cuda())
        self.output = nn.Linear(d_model, d_model).cuda()
        self.dropout = nn.Dropout(p=dropout)
        self.head = head

    def forward(self, query, key, value, mask=None):
        common = "mha{}".format(self.head)
        query_key = common + "_q"
        key_key = common + "_k"
        value_key = common + "_v"
        attention_key = common
        query = apply_quantization_activations(query_key, query)
        key = apply_quantization_activations(key_key, key)
        value = apply_quantization_activations(value_key, value)

        batch_size = query.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1)
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.heads, (query, key, value))]
        x, self.attn = ScaledDotProductQuant(query, key, value, self.dropout, mask)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        x = apply_quantization_activations(attention_key, x)
        x = self.output(x)
        return x


class ClassifierQuant(nn.Module):
    def __init__(self, d_model, n_class):
        super(ClassifierQuant, self).__init__()
        self.classifier = nn.Linear(d_model, n_class)

    def forward(self, x):
        key_name = "classifier"
        x = apply_quantization_activations(key_name, x)
        return self.classifier(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros((max_len, d_model), device="cuda")
        position = torch.arange(0, max_len).unsqueeze(1).cuda()
        scale = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)).cuda()
        pe[:, 0::2] = torch.sin(position * scale)
        pe[:, 1::2] = torch.cos(position * scale)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab, maxlen):
        super(Embeddings, self).__init__()
        self.token_embedding = nn.Embedding(vocab, d_model)
        self.pos_embedding = nn.Embedding(maxlen, d_model)
        self.d_model = d_model
        self.maxlen = maxlen

    def forward(self, x):
        positions = self.pos_embedding(torch.arange(start=0, end=self.maxlen, device="cuda"))[:x.size(1), :]
        tokens = self.token_embedding(x)
        return (positions + tokens) * math.sqrt(self.d_model)


class sublayerConnectionAttention(nn.Module):
    def __init__(self, h, d_model, dropout_head=0.1, dropout_connection=0.1, head=0):
        super(sublayerConnectionAttention, self).__init__()
        self.multiheads = MultiheadAttentionQuant(h, d_model, dropout_head, head)
        self.layernorm = LayerNormQuant(d_model)
        self.dropout = nn.Dropout(p=dropout_connection)

    def forward(self, x, mask=None):
        original = x
        x = self.layernorm(x)
        x = self.multiheads(x, x, x, mask)
        x = self.dropout(x)
        return x + original


class sublayerConnectionFFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout_ffn=0.1, dropout_connection=0.1, head=0):
        super(sublayerConnectionFFN, self).__init__()
        self.ffn = PositionalWiseFFNQuant(d_model, d_ff, dropout_ffn, head).cuda()
        self.layernorm = LayerNormQuant(d_model)
        self.dropout = nn.Dropout(p=dropout_connection)

    def forward(self, x):
        original = x
        x = self.layernorm(x)
        x = self.dropout(self.ffn(x))
        return x + original


class Model(nn.Module):
    def __init__(self,
                 n_class,
                 vocab,
                 n_layers=6,
                 h=2,
                 d_model=512,
                 d_ff=128,
                 maxlen=512,
                 dropout_encodings=0.1,
                 dropout_connection_attention=0.1,
                 dropout_connection_ffn=0.1,
                 dropout_attention=0.1,
                 dropout_ffn=0.1):
        super(Model, self).__init__()
        self.input_embeddings = Embeddings(d_model, vocab, maxlen)
        self.input_encodings = PositionalEncoding(d_model, dropout_encodings, maxlen)
        # self.layernorm = LayerNorm(d_model)
        self.sublayer_attention = list()
        self.sublayer_ffn = list()
        for i in range(n_layers):
            self.sublayer_attention.append(sublayerConnectionAttention(
                h, d_model, dropout_attention, dropout_connection_attention, i))
            self.sublayer_ffn.append(sublayerConnectionFFN(
                d_model, d_ff, dropout_ffn, dropout_connection_ffn, i))
        self.classifier = ClassifierQuant(d_model, n_class)
        self.n_layers = n_layers

        self.init_params()

    def forward(self, x, mask=None):
        embeddings = self.input_embeddings(x)
        encodings = self.input_encodings(embeddings)
        x = embeddings + encodings
        for i in range(self.n_layers):
            x = self.sublayer_attention[i](x, mask)
            x = self.sublayer_ffn[i](x)
            # x = self.layernorm(x)
        cls_repre = x[:, 0, :]
        outputs = self.classifier(cls_repre)
        return outputs

    def init_params(self, default_initialization=False):
        # Not mentioned in the paper, but other implementations used xavier.
        if not default_initialization:
            for name, p in self.named_parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
