from quantization.binarize import IRLinear
import torch.nn as nn

class AverageMeter(object):
    """
    Computes and stores the average and current value.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def accuracy(output, target, topk=(1,)):

    """
    Computes the precision@k for the specified values of k
    """ 
    maxk = max(topk) 
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True) 
    pred = pred.t() 
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []

    for k in topk:

        correct_k = correct[:k].view(-1).float().sum(0) 
        res.append(correct_k.mul_(100.0 / batch_size)) 
        
    return res


def change_t(model, t_value):
    """
    Recursively replace linear layers with binary layers
    """
    
    for name, layer in model.named_children():
        # change layer's t value
        if type(layer) == IRLinear:
            model.__dict__["_modules"][name].t = t_value
            
        else:
            layer_types = [type(layer) for layer in layer.modules()]

            if IRLinear in layer_types:
                change_t(layer, t_value)
    return

def module_copy(source_module, target_module):
    """
    Copy model weights from source_module to target_module
    """
    assert isinstance(source_module, nn.Module) and isinstance(target_module, nn.Module)
    try:
        target_module.load_state_dict(source_module.state_dict())
    except:
        raise Exception("Two models do not match!")
    return target_module
