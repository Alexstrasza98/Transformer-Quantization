import torch.nn as nn


def module_copy(source_module, target_module):
    assert isinstance(source_module, nn.Module) and isinstance(target_module, nn.Module)
    try:
        target_module.load_state_dict(source_module.state_dict())
    except:
        raise Exception("Two models do not match!")
    return target_module
