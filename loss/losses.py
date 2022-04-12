import torch
import torch.nn

def CrossEntropyLoss():
    return torch.nn.CrossEntropyLoss()
def MSE(gt,pre,eps=0.0001):
    return torch.sqrt(torch.pow(pre - gt, 2) + eps).mean()
