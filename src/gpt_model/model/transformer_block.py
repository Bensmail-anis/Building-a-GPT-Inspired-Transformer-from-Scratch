import torch.nn as nn
from .attention import MultiHeadAttention
from .feed_forward import FeedForward

class Block(nn.Module):
    """Transformer block : communication + computation"""
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout) # after the communication is done , each token must think about itself
        self.ln1 = nn.LayerNorm(n_embd) # the normalization layer with n_embd features
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # communication , ln1 is layer norm applied to the input before the attention layer , slight deviation from the paper
        x = x + self.ffwd(self.ln2(x)) # computation
        # residual connection implementation
        return x