# src/training/train.py
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from src.config.model_config import Config
from src.model.gpt import GPT
from src.model.utils import causal_mask
from tokenizers import Tokenizer
import math

# ---------------- Dataset ----------------
class TextDataset(Dataset):
    def __init__(self, token_ids, block_size):
        self.data = token_ids
        self.block_size = block_size

    def __len__(self):
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.block_size]
        y = self.data[idx + 1: idx + 1 + self.block_size]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

def collate_fn(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs), torch.stack(ys)

def load_tokenizer(path):
    return Tokenizer.from_file(path)

def tokenize_file(tokenizer, file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return tokenizer.encode(text).ids

# ---------------- Training ----------------
def train():
    cfg = Config
    device = cfg.device

    # ---- Load tokenizer and tokenize data ----
    tokenizer = load_tokenizer(cfg.TOKENIZER_PATH)
    token_ids = tokenize_file(tokenizer, cfg.DATA_PATH)

    # ---- Split train / validation ----
    split_idx = int(0.9 * len(token_ids))
    train_ids = token_ids[:split_idx]
    val_ids = token_ids[split_idx:]

    train_dataset = TextDataset(train_ids, cfg.max_seq_len)
    val_dataset = TextDataset(val_ids, cfg.max_seq_len)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)

    # ---- Initialize model ----
    model = GPT(cfg).to(device)
    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate)

    start_epoch = 0
    # ---- Resume from checkpoint if exists ----
    if os.path.exists(cfg.MODEL_PATH):
        checkpoint = torch.load(cfg.MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint)
        print(f"Resumed model from {cfg.MODEL_PATH}")

    model.train()
    for epoch in range(start_epoch, cfg.num_epochs):
        total_loss = 0.0
        for step, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # causal mask
            mask = causal_mask(x_batch.size(0), x_batch.size(1), device)

            logits, _ = model(x_batch, attention_mask=mask)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, cfg.vocab_size), y_batch.view(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            total_loss += loss.item()
            if step % 50 == 0:
                print(f"Epoch {epoch} Step {step} Train Loss: {loss.item():.4f}")

        avg_train_loss = total_loss / (step + 1)
        train_ppl = math.exp(avg_train_loss)
        print(f"Epoch {epoch} Average Train Loss: {avg_train_loss:.4f} | Train Perplexity: {train_ppl:.2f}")

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                mask = causal_mask(x_val.size(0), x_val.size(1), device)
                logits, _ = model(x_val, attention_mask=mask)
                loss = torch.nn.functional.cross_entropy(logits.view(-1, cfg.vocab_size), y_val.view(-1))
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        val_ppl = math.exp(avg_val_loss)
        print(f"Epoch {epoch} Validation Loss: {avg_val_loss:.4f} | Validation Perplexity: {val_ppl:.2f}")
        model.train()

        # ---- Save checkpoint ----
        os.makedirs(os.path.dirname(cfg.MODEL_PATH), exist_ok=True)
        torch.save(model.state_dict(), cfg.MODEL_PATH)
        print(f"Saved checkpoint to {cfg.MODEL_PATH}")

if __name__ == "__main__":
    train()
