import torch
from torch import nn, optim
import torch.nn.functional as F
import math


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
 
    def forward(self, q, k, v, mask=None):
        d_k = k.size(-1) # get the size of the key
        attn = torch.bmm(q, k.transpose(1, 2)) 
        # we get an attention score between each position in the sequence
        # for each batch
        attn = attn / math.sqrt(d_k)
        # mask attention weights if necessary
        if mask is not None: attn = attn.masked_fill(mask, 0)
        attn = F.softmax(attn, dim = 2)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v) 
        return output
    
class AttentionHead(nn.Module):
    """A single attention head"""
    def __init__(self, d_model, d_feature, dropout=0.1):
        super().__init__()
        ##linear transformation layer for the keys, values and queries before attention layer
        self.q_layer = nn.Linear(d_model, d_feature)
        self.k_layer = nn.Linear(d_model, d_feature)
        self.v_layer = nn.Linear(d_model, d_feature)
        self.attn = ScaledDotProductAttention(dropout)
    def forward(self, queries, keys, values, mask=None):
        q = self.q_layer(queries) 
        k = self.k_layer(keys) 
        v = self.v_layer(values) 
        x = self.attn(q, k, v)
        return x