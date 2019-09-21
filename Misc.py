import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
    
        self.size = d_model
        
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        
        self.eps = eps
    
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm
    
class PosEnc(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()        
        # compute positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        factor = 10000**(torch.arange(0, d_model, 2)/d_model).float()
                             
        pe[:, 0::2] = torch.sin(position * factor)
        pe[:, 1::2] = torch.cos(position * factor)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=False)
         
    def forward(self, x):
        ##returns positional encodings for everything in the batch
        print(i.size(0) for i in x)
        return torch.cat([self.weight[:, :i.size(0), :] for i in x], 0)