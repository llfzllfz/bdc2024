import torch
import torch.nn as nn
from Transformer_EncDec import Encoder, EncoderLayer
from SelfAttention_Family import FullAttention, AttentionLayer, TemporalAttention, Sptial_Attention
from Embed import DataEmbedding_inverted


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
        window_lists = [42, 168]
        self.encoder_sp1 = Encoder(
            [
                EncoderLayer(
                    Sptial_Attention(dim = 64, dropout = 0.1, C = 5, n = 9),
                    64,
                    64,
                    0.1,
                    activation='gelu'
                ) for l in range(1)
            ],
            norm_layer=torch.nn.LayerNorm(64)
        )
        self.encoder_sp2 = Encoder(
            [
                EncoderLayer(
                    Sptial_Attention(dim = 64, dropout = 0.1, C = 5, n = 5),
                    64,
                    64,
                    0.1,
                    activation='gelu'
                ) for l in range(1)
            ],
            norm_layer=torch.nn.LayerNorm(64)
        )
        self.encoder_sp3 = Encoder(
            [
                EncoderLayer(
                    Sptial_Attention(dim = 64, dropout = 0.1, C = 5, n = 5),
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
        # Decoder
        self.projection = nn.Linear(64, 24, bias=True)

    def forecast(self, x1, x2):

        B, L, C, W = x1.size()

        x_enc = torch.cat([x1.view(B, L, -1), x2], dim = -1)

        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, _, N = x_enc.shape

        enc_out = self.enc_embedding(x_enc, None) # B, L, 9, 64 # B, 4, 9, C

        label = enc_out[:, -1, :].unsqueeze(1)
        enc_out = enc_out[:, :C * W, :].view(B, C, W, -1)

        enc_out1, attns = self.encoder_sp1(enc_out[:, :, W // 2, :], x2 = enc_out, attn_mask=None) # B, L, 64 # B, 4, L
        enc_out2, attns = self.encoder_sp2(enc_out[:, :, W // 2, :], x2 = enc_out[:, :, [0, 2, 4, 6, 8], :], attn_mask = None)
        enc_out3, attns = self.encoder_sp3(enc_out[:, :, W // 2, :], x2 = enc_out[:, :, [1, 3, 4, 5, 7], :], attn_mask = None)

        enc_out1 = torch.cat([enc_out1, enc_out2, enc_out3, label], dim = 1)

        enc_out1, attns = self.encoder2(enc_out1)

        dec_out = self.projection(enc_out1).permute(0, 2, 1)[:, :, -1]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, -1].unsqueeze(1))
        dec_out = dec_out + (means[:, 0, -1].unsqueeze(1))
        return dec_out

    def forward(self, x1, x2):
        dec_out = self.forecast(x1, x2)
        return dec_out