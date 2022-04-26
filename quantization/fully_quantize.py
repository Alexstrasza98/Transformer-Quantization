# quantization: weight & activations
# let's start with weight quantization
import torch


def quantization(X, xmin, xmax, k):
    s = (xmax - xmin) / (2 ** k - 1)
    q = torch.div(torch.clamp(X, min=xmin, max=xmax), s, rounding_mode="floor") * s + xmin
    return q


if __name__ == "__main__":
    X = torch.rand((10, ))
    xmin = 0
    xmax = 1
    k = 8
    print(quantization(X, xmin, xmax, k))
