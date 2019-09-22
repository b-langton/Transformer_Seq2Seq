import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
    
        self.size = d_model
        
        # create two learnable parameters to calibrate normalisation
        self.a = nn.Parameter(torch.ones(self.size))
        self.b = nn.Parameter(torch.zeros(self.size))
        
        self.e = eps
    
    def forward(self, x):
        norm = self.a * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.e) + self.b
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
    
def nopeak_mask(n, opt):
    mask = np.triu(np.ones((1, n, n)),
    k=1).astype('uint8')
    mask =  Variable(torch.from_numpy(mask) == 0)
    return mask

def create_masks(src, trg, opt):
    src_mask = (src != opt.src_pad).unsqueeze(-2)
    if trg is not None:
        trg_mask = (trg != opt.trg_pad).unsqueeze(-2)
        size = trg.size(1) # get seq_len for matrix
        np_mask = nopeak_mask(size, opt)
        trg_mask = trg_mask & np_mask
        
    else:
        trg_mask = None
    return src_mask, trg_mask