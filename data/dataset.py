from datasets import load_dataset
from tokenizers import Tokenizer
import torch
from torch.utils.data import Dataset, DataLoader

class C4Dataset(Dataset):
    def __init__(self, split="train", seq_len=512, sample_size=1000):
        self.dataset = load_dataset("allenai/c4", "en", split=split, streaming=True).take(sample_size)
        self.tokenizer = Tokenizer.from_pretrained("gpt2")
        self.seq_len = seq_len

    def __len__(self):
        return 1000  # Hardcoded for streaming mode

    def __getitem__(self, idx):
        sample = next(iter(self.dataset.skip(idx).take(1)))
        tokens = self.tokenizer.encode(sample["text"]).ids[:self.seq_len]
        tokens = tokens + [0] * (self.seq_len - len(tokens))  # Pad to seq_len
        return torch.tensor(tokens[:-1]), torch.tensor(tokens[1:])

def get_dataloader(batch_size=8, split="train"):
    dataset = C4Dataset(split=split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"))