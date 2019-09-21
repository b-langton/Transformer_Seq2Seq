import torch
from torch import nn, optim
import torch.nn.functional as F
from Encoder import Encoder
from Decoder import Decoder



class Transformer(nn.Module): 
    def __init__(self, n_blocks, d_model, d_feature, d_ff, dropout, d_vocab_in, d_vocab_out, max_seq_len = 512):
        '''n_blocks: number of encoder and decoder blocks. d_model: Size of word feature vector d_feature: size of attn head input feature
        d_ff: size between linear layers'''
        super(Transformer,self).__init__()
        self.encoder = Encoder(n_blocks, d_model, d_feature, d_ff,dropout, max_seq_len = 512)
        self.decoder = Decoder(n_blocks, d_model, d_feature, d_ff,dropout, max_seq_len = 512)
        self.encoder_embed = nn.Embedding(d_vocab_in, d_model)
        self.decoder_embed = nn.Embedding(d_vocab_out, d_model)
        self.linear = nn.Linear(d_model, d_vocab_out)
    def forward(self, src, tgt, src_mask, tgt_mask):
        enc_in = self.encoder_embed(src)
        tgt = self.decoder_embed(tgt)
        enc_out = self.encoder(enc_in)
        out = self.decoder(tgt, enc_out, src_mask, tgt_mask)
        out = F.softmax(self.linear(out), dim = 2)
        return out