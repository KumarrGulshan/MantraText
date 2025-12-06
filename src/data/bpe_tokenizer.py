from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteDecoder
import os

class BPETokenizer:
    def __init__(self, tokenizer_path="data/tokenizer.json"):
        self.tokenizer_path = tokenizer_path

        if os.path.exists(tokenizer_path):
            self.tokenizer = Tokenizer.from_file(tokenizer_path)
        else:
            self.tokenizer = None

    def train(self, corpus_path, vocab_size=8000):
        print("Training ByteLevel BPE tokenizer...")

        tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
        tokenizer.decoder = ByteDecoder()

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=2,
            special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"]
        )

        tokenizer.train([corpus_path], trainer)

        # Save tokenizer as single JSON file
        os.makedirs("data", exist_ok=True)
        tokenizer.save(self.tokenizer_path)

        self.tokenizer = tokenizer

        print(f"Tokenizer saved to {self.tokenizer_path}")

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def decode(self, ids):
        return self.tokenizer.decode(ids)


if __name__ == "__main__":
    corpus = "data/samples/corpus.txt"
    tok = BPETokenizer()
    tok.train(corpus, vocab_size=8000)

    test = "Never stop learning new things."
    ids = tok.encode(test)
    print("Encoded:", ids)
    print("Decoded:", tok.decode(ids))
