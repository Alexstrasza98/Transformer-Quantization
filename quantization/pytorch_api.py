import torch
from .transformer_raw import *

# test with quantization on the feed-forward network
# test with quantization aware training
# fusing model when training/evaluation???

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


def quantization_training(d_model, d_ff, dropout_ffn, fused_modules):
    model = PositionalWiseFFNQuantization(d_model, d_ff, dropout_ffn)
    model.train()
    model.qconfig = torch.quantization.get_default_qat_qconfig("fbgemm")
    model_fused = torch.quantization.fuse_modules(model, fused_modules)
    model_prepared = torch.quantization.prepare_qat(model_fused)
    return model_prepared


class sublayerConnectionFFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout_ffn=0.1, dropout_connection=0.1):
        super(sublayerConnectionFFN, self).__init__()
        self.ffn = quantization_training(d_model, d_ff, dropout_ffn, fused_modules=[["w_1", "relu1"]]).cuda()
        self.layernorm = LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout_connection)

    def forward(self, x):
        original = x
        x = self.layernorm(x)
        x = self.dropout(self.ffn(x))
        return x + original


class ModelQuant(nn.Module):
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
        super(ModelQuant, self).__init__()
        self.input_embeddings = Embeddings(d_model, vocab, maxlen)
        self.input_encodings = PositionalEncoding(d_model, dropout_encodings, maxlen)
        # self.layernorm = LayerNorm(d_model)
        self.sublayer_attention = list()
        self.sublayer_ffn = list()
        for _ in range(n_layers):
            self.sublayer_attention.append(sublayerConnectionAttention(
                h, d_model, dropout_attention, dropout_connection_attention))
            self.sublayer_ffn.append(sublayerConnectionFFN(
                d_model, d_ff, dropout_ffn, dropout_connection_ffn))
        self.classifier = Classifier(d_model, n_class)
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
