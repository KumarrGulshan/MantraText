# import torch
# from src.model.transformer_model import Transformer
# from src.data.bpe_tokenizer import BPETokenizer
# from src.config.model_config import Config
# import os

# class Generator:
#     def __init__(self, checkpoint_path):
#         print(f"Using device: {Config.device}")
#         # Load tokenizer
#         self.tokenizer = BPETokenizer(
#             vocab_path=Config.TOKENIZER_PATH,
#             merges_path=Config.MERGES_PATH,
#         )
#         vocab_size = self.tokenizer.tokenizer.get_vocab_size()

#         # Build Transformer
#         self.model = Transformer(
#             vocab_size=vocab_size,
#             embed_dim=Config.embed_dim,
#             num_heads=Config.n_heads,
#             ff_dim=Config.ffn_dim,
#             num_encoder_layers=Config.n_layers,
#             num_decoder_layers=Config.n_layers,
#             max_len=Config.max_seq_len,
#         ).to(Config.device)

#         # Load checkpoint
#         print(f"📂 Loading checkpoint: {checkpoint_path}")
#         state_dict = torch.load(checkpoint_path, map_location=Config.device)
#         self.model.load_state_dict(state_dict)
#         self.model.eval()
#         print("✅ Model loaded successfully.\n")
#         self.device = Config.device

#     def generate(self, prompt, max_tokens=50, temperature=1.0, top_k=50, top_p=0.9):
#         # Encode prompt
#         input_ids = self.tokenizer.encode(prompt)
#         input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)

#         # Decoder starts with <bos>
#         bos_id = self.tokenizer.tokenizer.token_to_id("<bos>")
#         eos_id = self.tokenizer.tokenizer.token_to_id("<eos>")
#         decoder_input = torch.tensor([[bos_id]], dtype=torch.long).to(self.device)

#         for _ in range(max_tokens):
#             logits = self.model(input_tensor, decoder_input)  # [1, seq_len, vocab]
#             next_token_logits = logits[0, -1] / temperature  # apply temperature

#             # Top-k filtering
#             if top_k > 0:
#                 topk_vals, topk_idx = torch.topk(next_token_logits, top_k)
#                 mask = next_token_logits < topk_vals[-1]
#                 next_token_logits[mask] = -float('Inf')

#             # Top-p (nucleus) filtering
#             if top_p < 1.0:
#                 sorted_logits, sorted_idx = torch.sort(next_token_logits, descending=True)
#                 cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
#                 sorted_idx_to_remove = cumulative_probs > top_p
#                 sorted_idx_to_remove[1:] = sorted_idx_to_remove[:-1].clone()
#                 sorted_idx_to_remove[0] = 0
#                 indices_to_remove = sorted_idx[sorted_idx_to_remove]
#                 next_token_logits[indices_to_remove] = -float('Inf')

#             # Sample next token
#             probs = torch.softmax(next_token_logits, dim=-1)
#             next_token = torch.multinomial(probs, num_samples=1)
#             decoder_input = torch.cat([decoder_input, next_token.unsqueeze(0)], dim=1)

#             # Stop if EOS
#             if next_token.item() == eos_id:
#                 break

#         # Remove BOS token
#         generated_ids = decoder_input[0].tolist()[1:]
#         return self.tokenizer.decode(generated_ids)


# if __name__ == "__main__":
#     generator = Generator("models/checkpoints/model.pt_epoch6.pt")
#     prompt = "The sun rises ?"
#     output = generator.generate(prompt, max_tokens=50, temperature=0.8, top_k=50, top_p=0.9)
#     print(f"\n📝 Prompt: {prompt}")
#     print(f"🤖 Generated: {output}")


# src/inference/generate.py
import torch
import torch.nn.functional as F
from src.config.model_config import Config
from src.model.gpt import GPT
from tokenizers import Tokenizer
from src.model.utils import causal_mask


def top_k_logits(logits, k):
    """ Keep only top-k logits, rest → -inf """
    if k == 0:
        return logits
    top_vals, _ = torch.topk(logits, k)
    min_val = top_vals[:, -1].unsqueeze(-1)
    return torch.where(logits < min_val, torch.full_like(logits, -1e10), logits)


def clean_decode(tokenizer, ids):
    """
    Improved decoding for clean output:
    - removes special tokens
    - merges BPE subwords (##, @@, Ġ, etc.)
    - fixes spacing
    """
    text = tokenizer.decode(ids, skip_special_tokens=True)

    # Fix common BPE artifacts
    text = text.replace("@@ ", "")        # sentencepiece/BPE continuation
    text = text.replace(" ##", "")        # wordpiece continuation
    text = text.replace("Ġ", " ")         # GPT2-space symbol

    # Remove multiple spaces
    text = " ".join(text.split())

    return text


def generate(
    prompt,
    tokenizer_path,
    model_path,
    max_new_tokens=50,
    temperature=1.0,
    top_k=50,
    device=None
):
    """ Simple GPT-style text generation """
    device = device or Config.device
    cfg = Config

    # 1️⃣ Load tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_path)

    # 2️⃣ Load model
    model = GPT(cfg).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 3️⃣ Encode input prompt
    input_ids = torch.tensor(
        [tokenizer.encode(prompt).ids],
        dtype=torch.long,
        device=device
    )

    # 4️⃣ Generation loop
    for _ in range(max_new_tokens):
        seq_len = input_ids.size(1)

        if seq_len > cfg.max_seq_len:
            input_ids = input_ids[:, -cfg.max_seq_len:]

        mask = causal_mask(input_ids.size(0), input_ids.size(1), device)

        logits, _ = model(input_ids, attention_mask=mask)

        logits = logits[:, -1, :] / max(temperature, 1e-8)
        logits = top_k_logits(logits, top_k)

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)

        input_ids = torch.cat([input_ids, next_id], dim=1)

    # 5️⃣ Clean decode
    output_ids = input_ids[0].tolist()
    return clean_decode(tokenizer, output_ids)


# ---------------- Example usage ----------------
if __name__ == "__main__":
    tokenizer_path = "data/tokenizer.json"
    model_path = "models/checkpoints/gpt_small.pt"
    prompt = "how are you?"

    generated_text = generate(
        prompt,
        tokenizer_path=tokenizer_path,
        model_path=model_path,
        max_new_tokens=50,
        temperature=0.8,
        top_k=50
    )

    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")
