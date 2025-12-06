# src/model/decoder_block.py
import torch.nn as nn
from src.model.multihead_attention import MultiHeadAttention

class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout=dropout)

        self.ln2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Pre-LN
        x_norm = self.ln1(x)
        attn_out, attn_weights = self.attn(x_norm, mask=mask)
        x = x + self.dropout(attn_out)

        x_norm = self.ln2(x)
        ff_out = self.ff(x_norm)
        x = x + self.dropout(ff_out)

        return x, attn_weights
