# quantization: weight & activations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from collections import defaultdict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

keys = {"scale_product_head_0",
        "scale_product_head_0_q",
        "scale_product_head_0_k",
        "scale_product_head_0_v",
        "mha0_q",
        "mha0_v",
        "mha0_k",
        "mha0",
        "ffn0_input",
        "ffn0_output",
        "classifier"}


class EMA_Activation:
    def __init__(self, mu=0.9):
        self.mu = mu
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = [val[0].clone(), val[1].clone()]

    def __call__(self, name, x):
        assert name in self.shadow
        if len(x.size()) < 4:
            if self.shadow[name][0] * self.shadow[name][1] == 0:
                new_xmin = x.min()
                new_xmax = x.max()
            else:
                new_xmin = (1.0 - self.mu) * x.min() + self.mu * self.shadow[name][0]
                new_xmax = (1.0 - self.mu) * x.max() + self.mu * self.shadow[name][1]
        else:
            num_heads = x.size(1)
            x_copy = x.clone().permute(0, 2, 3, 1).contiguous().view(num_heads, -1)
            new_xmin = x_copy.min(dim=1, keepdim=True)[0].unsqueeze(-1)
            new_xmax = x_copy.max(dim=1, keepdim=True)[0].unsqueeze(-1)
        self.shadow[name] = [new_xmin.clone(), new_xmax.clone()]
        return [new_xmin, new_xmax]


ema_activation = EMA_Activation()
for key in keys:
    if key.startswith("scale_product_head_0"):
        ema_activation.register(key, [torch.zeros(8, 1, 1, device=device), torch.zeros(8, 1, 1, device=device)])
    else:
        ema_activation.register(key, [torch.tensor(0), torch.tensor(0)])


class EMA_Weight:
    def __init__(self, mu=0.9):
        self.mu = mu
        self.shadow = defaultdict(float)

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def __call__(self, name, x, k=8):
        assert name in self.shadow
        xmax = x.max().item()
        xmin = x.min().item()
        s = (xmax - xmin) / (2 ** k - 1)
        q = torch.div(x, s, rounding_mode="floor") * s + xmin
        new_average = torch.round(q)
        self.shadow[name] = new_average.clone()
        return new_average


class LayerNorm(nn.Module):
    "Construct a layernorm module"

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features, device="cuda"))
        self.b_2 = nn.Parameter(torch.zeros(features, device="cuda"))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


def quantization_activations(X, name, k=8):
    if X.requires_grad:
        ema_activation(name, X)
    xmin, xmax = ema_activation.shadow[name]
    s = (xmax - xmin) / (2 ** k - 1)
    q = torch.div(torch.clamp(X, min=xmin, max=xmax), s, rounding_mode="trunc") * s + xmin
    q = torch.round(q)
    return q


"""
Does activations have backpropagation?
"""
class Quantization_Activations(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, name="", k=8):
        return quantization_activations(X, name, k)

    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs, None, None


_quantization_activations = Quantization_Activations.apply


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
        x = _quantization_activations(x, key_input)

        x = self.dropout(F.relu(self.w_1(x)))
        x = _quantization_activations(x, key_output)
        return self.w_2(x)


def ScaledDotProductQuant(query, key, values, dropout=None, mask=None, head=0):
    common = "scale_product_head_{}".format(head)
    query_key = common + "_q"
    key_key = common + "_k"
    value_key = common + "_v"
    attention_key = common
    query = _quantization_activations(query, query_key)
    key = _quantization_activations(key, key_key)
    values = _quantization_activations(values, value_key)

    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        mask = mask.squeeze(1)
        scores = scores.masked_fill_(mask == 0, -1e-9)
    p_atten = F.softmax(scores, dim=-1)
    if dropout:
        p_atten = dropout(p_atten)
    p_atten = _quantization_activations(p_atten, attention_key)
    return torch.matmul(p_atten, values), p_atten


class MultiheadAttentionQuant(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, head=0):
        super(MultiheadAttentionQuant, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.heads = list()
        self.attn = None
        for _ in range(3):
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
        query = _quantization_activations(query, query_key)
        key = _quantization_activations(key, key_key)
        value = _quantization_activations(value, value_key)

        batch_size = query.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1)
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.heads, (query, key, value))]
        x, self.attn = ScaledDotProductQuant(query, key, value, self.dropout, mask)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        x = _quantization_activations(x, attention_key)
        x = self.output(x)
        return x


class ClassifierQuant(nn.Module):
    def __init__(self, d_model, d_hidden, n_class):
        super(ClassifierQuant, self).__init__()
        self.hidden = nn.Linear(d_model, d_hidden)
        self.classifier = nn.Linear(d_hidden, n_class)

    def forward(self, x):
        key_name = "classifier"
        x = _quantization_activations(x, key_name)
        x = self.hidden(x)
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
        self.layernorm = LayerNorm(d_model)
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
        self.layernorm = LayerNorm(d_model)
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
                 d_hidden=1024,
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
        self.sublayer_attention = nn.ModuleList()
        self.sublayer_ffn = nn.ModuleList()
        for i in range(n_layers):
            self.sublayer_attention.append(sublayerConnectionAttention(
                h, d_model, dropout_attention, dropout_connection_attention, i))
            self.sublayer_ffn.append(sublayerConnectionFFN(
                d_model, d_ff, dropout_ffn, dropout_connection_ffn, i))
        self.classifier = ClassifierQuant(d_model, d_hidden, n_class)
        self.n_layers = n_layers
        self.ema_weight = None

        self.init_params()
        self.ema_init()

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

    def ema_init(self):
        self.ema_weight = EMA_Weight()
        for name, params in self.named_parameters():
            if params.requires_grad and "bias" not in name:
                self.ema_weight.register(name, params)

    def apply_ema(self, k=8):
        if not self.ema_weight:
            self.ema_init()
        for name, params in self.named_parameters():
            if params.requires_grad and "bias" not in name:
                self.ema_weight(name, params, k)
