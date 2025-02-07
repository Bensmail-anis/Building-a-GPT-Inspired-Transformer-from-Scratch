import torch

class TextDataset:
    def __init__(self, config):
        self.config = config
        self.chars = None
        self.vocab_size = None
        self.stoi = None
        self.itos = None
        self.train_data = None
        self.val_data = None

    def load_data(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # here are all the unique characters that occur in this text
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        
        # create a mapping from characters to integers
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        
        # Train and test splits
        data = torch.tensor(self.encode(text), dtype=torch.long)
        n = int(0.9 * len(data))  # first 90% will be train, rest val
        self.train_data = data[:n]
        self.val_data = data[n:]

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, l):
        return ''.join([self.itos[i] for i in l])

    def get_batch(self, split):
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - self.config.block_size, (self.config.batch_size,))
        x = torch.stack([data[i:i+self.config.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.config.block_size+1] for i in ix])
        x, y = x.to(self.config.device), y.to(self.config.device)
        return x, y