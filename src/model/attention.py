# src/model/attention.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        """
        Q, K, V: [batch, heads, seq, head_dim]
        mask: broadcastable to [batch, heads, seq, seq] (1 = keep, 0 = mask)
        """
        # q @ k^T -> [batch, heads, seq, seq]
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            # mask should be boolean or 0/1
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)  # [batch, heads, seq, head_dim]
        return out, attn
