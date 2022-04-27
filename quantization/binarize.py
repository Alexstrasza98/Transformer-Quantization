import math
import torch
import torch.nn as nn
from torch.autograd import Function


class BinaryLinearFunction(Function):
    """
    Implements binarization function for linear layer with Straight-Through Estimation (STE)
    """

    @staticmethod
    def forward(ctx, input, weight, bias=None):
        # binarize weights by sign function
        weight_mask = (weight > 1) | (weight < -1)
        weight = torch.sign(weight)
        
        # save for grad computing
        ctx.save_for_backward(input, weight, weight_mask, bias)
        
        # linear layer
        output = input.matmul(weight.t())
        if bias is not None:
            output += bias

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # retrieve saved variables
        input, weight, weight_mask, bias = ctx.saved_variables
        
        # computing grads
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.matmul(weight)

        if ctx.needs_input_grad[1]:
            # if weights' absolute value larger than 1, no grads
            grad_weight = grad_output.transpose(-1, -2).matmul(input)
            grad_weight.masked_fill_(weight_mask, 0.0)

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias


class BinarizedLinear(nn.Module):
    """
    Implements Binarization Layer using Binarization function
    """

    def __init__(self, in_features, out_features, bias=True):
        super(BinarizedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.normal_(0, 1 * (math.sqrt(1.0 / self.in_features)))
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, input):
        if self.bias is not None:
            return BinaryLinearFunction.apply(input, self.weight, self.bias)
        else:
            raise Exception

    def __repr__(self):
        return self.__class__.__name__ + " (" + str(self.in_features) + " -> " + str(self.out_features) + ")"


def binarize(model, binarize_all_linear=False):
    """
    Recursively replace linear layers with binary layers
    :param model: Model to be binarized
    :param binarize_all_linear: Binarize all layers
    :return: Binarized model
    """

    for name, layer in model.named_children():
        # Binarization
        if type(layer) == nn.Linear and binarize_all_linear:
            model.__dict__["_modules"][name] = BinarizedLinear(layer.in_features, layer.out_features)
        elif type(layer) == nn.Linear:
            if name in ["0", "1", "2"]:
                model.__dict__["_modules"][name] = BinarizedLinear(layer.in_features, layer.out_features)
        else:
            layer_types = [type(layer) for layer in layer.modules()]

            if nn.Linear in layer_types:
                binarize(layer, binarize_all_linear)

    return model