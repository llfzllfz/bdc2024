import torch
import torch.nn as nn
from Transformer_EncDec import Encoder, EncoderLayer
from SelfAttention_Family import FullAttention, AttentionLayer, TemporalAttention, Sptial_Attention
from Embed import DataEmbedding_inverted
from Dec import Multi_CNN_Decoder

class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, device):
        super(Model, self).__init__()
        self.seq_len = 168
        self.pred_len = 24
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(168, 64, 0.1)
        # Encoder
        self.encoder1 = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, attention_dropout=0.1,
                                      output_attention=True), 64, 1),
                    64,
                    64,
                    0.1,
                    activation='gelu'
                ) for l in range(1)
            ],
            norm_layer=torch.nn.LayerNorm(64)
        )
        self.encoder = Encoder(
            [
                EncoderLayer(
                    Sptial_Attention(dim = 64, dropout = 0.1),
                    64,
                    64,
                    0.1,
                    activation='gelu'
                ) for l in range(1)
            ],
            norm_layer=torch.nn.LayerNorm(64)
        )
        self.encoder2 = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, attention_dropout=0.1,
                                      output_attention=True), 64, 1),
                    64,
                    64,
                    0.1,
                    activation='gelu'
                ) for l in range(1)
            ],
            norm_layer=torch.nn.LayerNorm(64)
        )

        self.projection = Multi_CNN_Decoder()

    def forecast(self, x1, x2, x3 = None, x4 = None):

        B, L, H, W = x1.size()

        x_enc = torch.cat([x1.view(B, L, -1), x2], dim = -1)

        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, _, N = x_enc.shape

        enc_out = self.enc_embedding(x_enc, None) # B, L, 9, 64 # B, 4, 9, C

        label = enc_out[:, -1, :].unsqueeze(1)
        enc_out = enc_out[:, :H * W, :]

        enc_out, attns = self.encoder1(enc_out)
        enc_out = enc_out.view(B, H, W, -1)
        
        enc_out, attns = self.encoder(enc_out[:, :, H // 2, :], x2 = enc_out, attn_mask=None) # B, L, 64 # B, 4, L

        enc_out = torch.cat([enc_out, label], dim = 1)

        enc_out, attns = self.encoder2(enc_out)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, -1]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, -1].unsqueeze(1))
        dec_out = dec_out + (means[:, 0, -1].unsqueeze(1))
        return dec_out

    def forward(self, x1, x2, x3 = None, x4 = None):
        dec_out = self.forecast(x1, x2, x3, x4)
        return dec_out