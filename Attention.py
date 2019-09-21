import torch
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F
import math
from AttentionSublayers import AttentionHead
 
class MultiHeadAttention(nn.Module):
    
    def __init__(self, d_model, d_feature, dropout=0.1):
        ## d_model must be divisible by d_feature
        super().__init__()
        self.d_model = d_model
        self.d_feature = d_feature
        self.n_heads = d_model//d_feature
        
        
 
       
        self.attn_heads = nn.ModuleList([
            AttentionHead(d_model, d_feature, dropout) for i in range(self.n_heads)
        ])
        self.projection = nn.Linear(d_model, d_model) 
     
    def forward(self, queries, keys, values, mask=None):
        x = [attn(queries, keys, values, mask=mask) 
             for i, attn in enumerate(self.attn_heads)]
        # reconcatenate
        x = torch.cat(x, dim=2) # (Batch, Seq, D_Model)
        x = self.projection(x) 
        return x

 
 