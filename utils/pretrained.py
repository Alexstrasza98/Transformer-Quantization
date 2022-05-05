import torch
from transformers import AutoTokenizer
from quantization.transformer import Transformer
from constants import *
from collections import OrderedDict
import io
import os
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_bert_weights(path="D://Downloads/pytorch_model.bin"):
    # load dataset
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # create model
    model = Transformer(4,
                        tokenizer.vocab_size,
                        BASELINE_MODEL_NUMBER_OF_LAYERS,
                        BASELINE_MODEL_NUMBER_OF_HEADS,
                        BASELINE_MODEL_DIM)

    # load pretrained bert?
    with open(path, 'rb') as f:
        buffer = io.BytesIO(f.read())
        pretrained_model = torch.load(buffer, map_location=device)
    f.close()
    params = OrderedDict()

    params_name = []
    for key in model.state_dict():
        if key.startswith("input_encodings.pe"):
            continue
        if key.startswith("sublayer_ffn"):
            break
        params_name.append(key)
    temp = []
    for key in pretrained_model:
        if key.startswith("bert.embeddings.token_type_embeddings"):
            continue
        if key.startswith("bert.embeddings.LayerNorm"):
            continue
        if key.startswith("bert.encoder.layer.0.intermediate"):
            continue
        if key.startswith("bert.encoder.layer.0.output.dense"):
            continue
        if key.startswith("bert.encoder.layer.0.output.LayerNorm"):
            continue
        if key.startswith("bert.encoder.layer.1.intermediate"):
            continue
        if key.startswith("bert.encoder.layer.1.output.dense"):
            continue
        if key.startswith("bert.encoder.layer.1.output.LayerNorm"):
            continue
        temp.append(pretrained_model[key])

    for i, weights in enumerate(temp):
        if i < len(params_name):
            params[params_name[i]] = weights
        else:
            break

    model.load_state_dict(params, strict=False)
    model.eval()
    if not os.path.exists("../pretrained_weights"):
        os.mkdir("../pretrained_weights")
    torch.save(model.state_dict(), "../pretrained_weights/pretrained_weights.pth")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="path to your pretrained checkpoint", default="D://Downloads/pytorch_model.bin")
    args = parser.parse_args()
    model = load_bert_weights(args.path)
