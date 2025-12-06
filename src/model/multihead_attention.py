# src/model/multihead_attention.py
import torch
import torch.nn as nn
from src.model.attention import ScaledDotProductAttention

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn = ScaledDotProductAttention(dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        # x: [batch, seq, embed_dim] -> [batch, heads, seq, head_dim]
        b, t, _ = x.size()
        x = x.view(b, t, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def _combine_heads(self, x):
        # x: [batch, heads, seq, head_dim] -> [batch, seq, embed_dim]
        x = x.transpose(1, 2).contiguous()
        b, t, _, _ = x.size()
        return x.view(b, t, self.embed_dim)

    def forward(self, x, mask=None):
        """
        x: [batch, seq, embed_dim]
        mask: either None or shape [batch, seq, seq] (causal) or broadcastable to [batch, heads, seq, seq]
        """
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = self._split_heads(Q)
        K = self._split_heads(K)
        V = self._split_heads(V)

        # prepare mask: expand to [batch, heads, seq, seq]
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # [batch, 1, seq, seq]
            # otherwise assume already broadcastable

        out, attn = self.attn(Q, K, V, mask=mask)
        out = self._combine_heads(out)
        out = self.out_proj(out)
        out = self.dropout(out)
        return out, attn
