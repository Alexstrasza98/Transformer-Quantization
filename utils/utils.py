import torch

# Count how many trainable weights the model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Count how much large memory this model uses
def count_memory_size(model):
    memory_dict = {torch.float32:4, torch.float64:8}
    return sum(p.numel()*memory_dict[p.data.dtype] for p in model.parameters() if p.requires_grad)