import torch
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
import random
import math
from Attention import MultiHeadAttention
from Misc import LayerNorm, PosEnc
class DecoderBlock(nn.Module):
    def __init__(self, d_model=512, d_feature=64,
                 d_ff=2048, dropout=0.1):
        super().__init__()
        self.n_heads = d_model//d_feature
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.masked_attn = MultiHeadAttention(d_model, d_feature, dropout)
        self.attn = MultiHeadAttention(d_model, d_feature, dropout)
        self.position_wise_ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)
         
    def forward(self, x, enc_out, 
                src_mask=None, tgt_mask=None):
        #apply masked attention with decoder inputs as keys, values, and queries
        att = self.masked_attn(x, x, x, mask=src_mask)
        #residual connection
        x += self.dropout(self.norm1(att))
        #apply masked attention with decoder inputs as queries and encoder outputs as keys and values
        att = self.attn(queries=x, keys=enc_out, values=enc_out, mask=tgt_mask)
        x += self.dropout(self.norm2(att))
        pos = self.position_wise_ff(x)
        x += self.dropout(self.norm3(pos))
        return x
    
class Decoder(nn.Module):
    def __init__(self, n_blocks=6, d_model=512, d_feature=64,
                 d_ff=2048, dropout=0.1, max_seq_len = 512):
        super().__init__()
        self.n_heads = d_model//d_feature
        self.pos_encoding = PosEnc(d_model, max_len = max_seq_len)
        self.decoders = nn.ModuleList([
            DecoderBlock(d_model=d_model, d_feature=d_feature,
                         d_ff=d_ff, dropout=dropout)
            for i in range(n_blocks)
        ])
         
    def forward(self, x: torch.FloatTensor, 
                enc_out: torch.FloatTensor, 
                src_mask=None, tgt_mask=None):
        x  = x + self.pos_encoding(x)
        for decoder in self.decoders:
            x = decoder(x, enc_out, src_mask=src_mask, tgt_mask=tgt_mask)
        return x