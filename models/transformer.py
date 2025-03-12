import torch.nn as nn
from .attention import GatedLinearAttention
from .norms import RMSNorm, SwiGLU

class GLATransformer(nn.Module):
    def __init__(self, vocab_size, dim, num_layers, num_heads):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "attn": GatedLinearAttention(dim, num_heads),
                "ffn": SwiGLU(dim),
                "norm1": RMSNorm(dim),
                "norm2": RMSNorm(dim)
            }) for _ in range(num_layers)
        ])
        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            # Attention
            attn_out = layer["attn"](layer["norm1"](x))
            x = x + attn_out
            # FFN
            ffn_out = layer["ffn"](layer["norm2"](x))
            x = x + ffn_out
        return self.head(self.norm(x))