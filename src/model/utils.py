# src/model/utils.py
import torch

def causal_mask(batch_size, seq_len, device):
    # returns mask with 1 where allowed (past & present), 0 for future
    mask = torch.tril(torch.ones((seq_len, seq_len), device=device, dtype=torch.uint8))
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)   # [batch, seq, seq]
    return mask
