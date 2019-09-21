import torch
from torch import nn, optim
import torch.nn.functional as F
from Attention import MultiHeadAttention
from Misc import LayerNorm, PosEnc

class EncoderBlock(nn.Module):
    def __init__(self, d_model=512, d_feature=64,
                 d_ff=2048, dropout=0.1):
        super().__init__()
        self.attn_head = MultiHeadAttention(d_model, d_feature,dropout)
        self.layer_norm1 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.position_wise_feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.layer_norm2 = LayerNorm(d_model)
         
    def forward(self, x, mask=None):
        att = self.attn_head(x, x, x, mask=mask)
        #residual connection
        x = x + self.dropout(self.layer_norm1(att))
        pos = self.position_wise_feed_forward(x)
        x = x + self.dropout(self.layer_norm2(pos))
        return x

class Encoder(nn.Module):
    def __init__(self, n_blocks=6, d_model=512, d_feature = 32,
                  d_ff=2048, dropout=0.1, max_seq_len = 512):
        super().__init__()
        self.n_heads = d_model//d_feature
        self.pos_encoding = PosEnc(d_model, max_len = max_seq_len)
        self.encoders = nn.ModuleList([
            EncoderBlock(d_model=d_model, d_feature=d_feature,
                         d_ff=d_ff, dropout=dropout)
            for _ in range(n_blocks)
        ])
     
    def forward(self, x: torch.FloatTensor, mask=None):
        x += self.pos_encoding(x)
        for encoder in self.encoders:
            x = encoder(x)
        return x