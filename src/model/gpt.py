# src/model/gpt.py
import torch
import torch.nn as nn
from src.model.decoder_block import DecoderBlock
from src.config.model_config import Config

class GPT(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.embed_dim)
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([
            DecoderBlock(config.embed_dim, config.n_heads, config.ffn_dim, dropout=config.dropout)
            for _ in range(config.n_layers)
        ])
        self.ln_f = nn.LayerNorm(config.embed_dim)

        # LM head (weight tying)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        # tie weights
        self.lm_head.weight = self.tok_emb.weight

        # initialize weights
        self._init_weights()

    def forward(self, idx, attention_mask=None):
        """
        idx: [batch, seq]
        attention_mask: causal mask [batch, seq, seq] or None
        """
        device = next(self.parameters()).device
        b, t = idx.size()
        assert t <= self.config.max_seq_len, "Sequence too long for model"

        positions = torch.arange(0, t, device=device).unsqueeze(0).expand(b, t)
        x = self.tok_emb(idx) + self.pos_emb(positions)
        x = self.drop(x)

        attn_weights = None
        for block in self.blocks:
            x, attn_weights = block(x, mask=attention_mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # [batch, seq, vocab]
        return logits, attn_weights

    def _init_weights(self):
        # small, safe initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
