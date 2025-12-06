
# src/config/model_config.py
import torch

class Config:
    # model
    vocab_size = 8000
    max_seq_len = 128
    embed_dim = 384        # smaller for 8GB; you can bump to 512 if you have room
    n_heads = 6
    n_layers = 6
    ffn_dim = 1536         # usually 4*embed_dim
    seq_len = 32
    dropout = 0.1

    # training
    batch_size = 8         # keep small on CPU
    num_epochs = 20
    learning_rate = 3e-4
    grad_clip = 1.0

    # data / paths
    DATA_PATH = "data/samples/corpus.txt"
    TOKENIZER_PATH = "data/tokenizer.json"   # full tokenizer file (tokenizers json)
    MERGES_PATH = "data/merges.txt"

    MODEL_PATH = "models/checkpoints/gpt_small.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


