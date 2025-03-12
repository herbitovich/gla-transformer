import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm_x = x.norm(2, dim=-1, keepdim=True)
        rms_x = norm_x * (x.shape[-1] ** -0.5)
        return self.scale * x / (rms_x + self.eps)

class SwiGLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w1 = nn.Linear(dim, 4 * dim)
        self.w2 = nn.Linear(dim, 4 * dim)
        self.w3 = nn.Linear(4 * dim, dim)
        
    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))