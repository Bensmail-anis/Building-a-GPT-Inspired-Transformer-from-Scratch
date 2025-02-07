import torch

class ModelConfig:
    def __init__(self):
        self.batch_size = 64  # how many independent sequences will we process in parallel?
        self.block_size = 256  # what is the maximum context length for predictions?
        self.max_iters = 5000
        self.eval_interval = 500
        self.learning_rate = 3e-4
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.eval_iters = 200
        self.n_embd = 384
        self.n_head = 6
        self.n_layer = 6
        self.dropout = 0.2
