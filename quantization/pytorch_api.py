from .transformer_raw import *
from torch.quantization.qconfig import QConfig
from torch.quantization.fake_quantize import FusedMovingAvgObsFakeQuantize
from torch.quantization.observer import MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver


# test with quantization on the feed-forward network
# test with quantization aware training
# fusing model when training/evaluation???


def set_bit_num(bits_activation=8, bits_weight=8):
    assert bits_activation <= 8
    assert bits_weight <= 8
    fused_per_channel_wt_fake_quant = FusedMovingAvgObsFakeQuantize.with_args(
        observer=MovingAveragePerChannelMinMaxObserver,
        quant_min=-2 ** (bits_weight-1),
        quant_max=2 ** (bits_weight-1) - 1,
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric)
    qconfig = QConfig(activation=FusedMovingAvgObsFakeQuantize.with_args(observer=MovingAverageMinMaxObserver,
                                                                         quant_min=0,
                                                                         quant_max=2 ** (bits_activation-1) - 1,
                                                                         reduce_range=True),
                      weight=fused_per_channel_wt_fake_quant)
    return qconfig


"""
class PositionalWiseFFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionalWiseFFN, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
"""


class PositionalWiseFFNQuantization(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionalWiseFFNQuantization, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.relu1 = nn.ReLU(inplace=False)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dequant = torch.quantization.DeQuantStub()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.quant(x)
        x = self.w_1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dequant(x)
        return x


def quantization_ffn_training(d_model, d_ff, dropout_ffn, fused_modules, k=(8, 8)):
    model = PositionalWiseFFNQuantization(d_model, d_ff, dropout_ffn)
    model.train()
    # model.qconfig = torch.quantization.get_default_qat_qconfig("fbgemm")
    model.qconfig = set_bit_num(k[0], k[1])
    model_fused = torch.quantization.fuse_modules(model, fused_modules)
    model_prepared = torch.quantization.prepare_qat(model_fused)
    return model_prepared


class sublayerConnectionFFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout_ffn=0.1, dropout_connection=0.1, quant_ffn=False):
        super(sublayerConnectionFFN, self).__init__()
        if quant_ffn:
            self.ffn = quantization_ffn_training(d_model, d_ff, dropout_ffn, fused_modules=[["w_1", "relu1"]]).cuda()
        else:
            self.ffn = PositionalWiseFFN(d_model, d_ff, dropout_ffn)
        self.layernorm = LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout_connection)

    def forward(self, x):
        original = x
        x = self.layernorm(x)
        x = self.dropout(self.ffn(x))
        return x + original


"""
class MultiheadAttention(nn.Module):
    '''
    Class of multi-head attention, break input features into h heads, 
    do attention, then concatenated together.
    Arguments:
    h       - number of heads
    d_model - dimensions of features per token throughout whole model
    dropout - dropout rate for output
    '''
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiheadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.heads = nn.ModuleList()
        self.attn = None
        for _ in range(3):
            self.heads.append(nn.Linear(d_model, d_model).to(device))
        self.output = nn.Linear(d_model, d_model).to(device)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1)
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.heads, (query, key, value))]
        x, self.attn = ScaledDotProduct(query, key, value, self.dropout, mask)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        x = self.output(x)
        return x
"""


def quantization_mha_training(h, d_model, dropout_mha, fused_modules=None, k=(8, 8)):
    model = MultiheadAttentionQuantization(h, d_model, dropout_mha)
    model.train()
    model.qconfig = set_bit_num(k[0], k[1])
    if fused_modules:
        model = torch.quantization.fuse_modules(model, fused_modules)
    model_prepared = torch.quantization.prepare_qat(model)
    return model_prepared


class MultiheadAttentionQuantization(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiheadAttentionQuantization, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.heads = nn.ModuleList()
        self.attn = None
        for _ in range(3):
            self.heads.append(nn.Linear(d_model, d_model).cuda())
        self.output = nn.Linear(d_model, d_model).cuda()
        self.dropout = nn.Dropout(p=dropout)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, query, key, value, mask=None):
        query = self.quant(query)
        key = self.quant(key)
        value = self.quant(value)  # optional
        batch_size = query.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1)
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.heads, (query, key, value))]
        x, self.attn = ScaledDotProduct(query, key, value, self.dropout, mask)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        x = self.output(x)
        x = self.dequant(x)
        return x


class sublayerConnectionAttention(nn.Module):
    def __init__(self, h, d_model, dropout_head=0.1, dropout_connection=0.1, quant_mha=False):
        super(sublayerConnectionAttention, self).__init__()
        if quant_mha:
            self.multiheads = quantization_mha_training(h, d_model, dropout_head)
        else:
            self.multiheads = MultiheadAttention(h, d_model, dropout_head)
        self.layernorm = LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout_connection)

    def forward(self, x, mask=None):
        original = x
        x = self.layernorm(x)
        x = self.multiheads(x, x, x, mask)
        x = self.dropout(x)
        return x + original


"""
class Classifier(nn.Module):
    '''Final classifier with one linear layer'''
    def __init__(self, d_model, d_hidden, n_class):
        super(Classifier, self).__init__()
        self.hidden = nn.Linear(d_model, d_hidden)
        self.classifier = nn.Linear(d_hidden, n_class)

    def forward(self, x):
        return self.classifier(F.relu(self.hidden(x)))

"""


def quantization_classifier_training(d_model, d_hidden, n_class, fused_modules=None, k=(8, 8)):
    model = ClassifierQuant(d_model, d_hidden, n_class)
    model.train()
    # model.qconfig = torch.quantization.get_default_qat_qconfig("fbgemm")
    model.qconfig = set_bit_num(k[0], k[1])
    if fused_modules:
        model = torch.quantization.fuse_modules(model, fused_modules)
    model_prepared = torch.quantization.prepare_qat(model)
    return model_prepared


class ClassifierQuant(nn.Module):
    def __init__(self, d_model, d_hidden, n_class):
        super(ClassifierQuant, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.hidden = nn.Linear(d_model, d_hidden)
        self.classifier = nn.Linear(d_hidden, n_class)
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.hidden(x)
        x = self.classifier(x)
        x = self.dequant(x)
        return x


class ModelQuant(nn.Module):
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
                 dropout_ffn=0.1,
                 quant_ffn=False,
                 quant_mha=False,
                 quant_classifier=False):
        super(ModelQuant, self).__init__()
        self.input_embeddings = Embeddings(d_model, vocab, maxlen)
        self.input_encodings = PositionalEncoding(d_model, dropout_encodings, maxlen)
        # self.layernorm = LayerNorm(d_model)
        self.sublayer_attention = nn.ModuleList()
        self.sublayer_ffn = nn.ModuleList()
        for _ in range(n_layers):
            self.sublayer_attention.append(sublayerConnectionAttention(
                h, d_model, dropout_attention, dropout_connection_attention, quant_mha))
            self.sublayer_ffn.append(sublayerConnectionFFN(
                d_model, d_ff, dropout_ffn, dropout_connection_ffn, quant_ffn))
        if quant_classifier:
            self.classifier = quantization_classifier_training(d_model, d_hidden, n_class)
        else:
            self.classifier = Classifier(d_model, d_hidden, n_class)
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
