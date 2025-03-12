import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedLinearAttention(nn.Module):
    def __init__(self, dim, num_heads, chunk_size=64):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.chunk_size = chunk_size
        
        self.Wq = nn.Linear(dim, dim)
        self.Wk = nn.Linear(dim, dim)
        self.Wv = nn.Linear(dim, dim)
        
        # Gating mechanism with low-rank bottleneck
        self.gate_proj = nn.Sequential(
            nn.Linear(dim, 16),  # Low-rank bottleneck
            nn.SiLU(),
            nn.Linear(16, num_heads * self.head_dim),
            nn.Sigmoid()
        )
        
        # Initialize parameters
        for layer in [self.Wq, self.Wk, self.Wv, self.gate_proj[0], self.gate_proj[2]]:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        B, L, _ = x.shape
        C = self.chunk_size
        H, D = self.num_heads, self.head_dim
        # Store original length before padding
        original_length = L
        
        # Pad sequence to chunk size
        if L % C != 0:
            padding = C - (L % C)
            x = F.pad(x, (0, 0, 0, padding))  # (batch, L_padded, dim)
        
        # Process chunks (now maintains padding internally)
        x_chunks = x.split(C, dim=1)
        outputs = []
        
        # Initialize hidden state
        S = torch.zeros(B, H, D, D, device=x.device)
        
        for chunk in x_chunks:
            q = self.Wq(chunk).view(B, -1, H, D).transpose(1, 2)  # [B, H, C, D]
            k = self.Wk(chunk).view(B, -1, H, D).transpose(1, 2)
            v = self.Wv(chunk).view(B, -1, H, D).transpose(1, 2)
            
            # Compute gating coefficients
            alpha = self.gate_proj(chunk).view(B, -1, H, D).transpose(1, 2)  # [B, H, C, D]
            alpha = 0.9 * alpha + 0.1  # Stabilize gates
            
            # Process chunk tokens sequentially
            chunk_output = []
            for t in range(C):
                q_t = q[:, :, t]  # [B, H, D]
                k_t = k[:, :, t]
                v_t = v[:, :, t]
                alpha_t = alpha[:, :, t]
                
                # Inter-chunk contribution
                inter = torch.einsum('bhd,bhde->bhe', q_t, S)  # [B, H, D] @ [B, H, D, D]
                
                # Intra-chunk attention
                intra = torch.einsum('bhd,bhkd->bhk', q_t, k[:, :, :t+1])  # [B, H, t+1]
                intra = F.softmax(intra / (D ** 0.5), dim=-1)
                intra = torch.einsum('bhk,bhke->bhe', intra, v[:, :, :t+1])
                
                # Combine outputs
                out_t = inter + intra
                chunk_output.append(out_t.unsqueeze(2))
                
                # Update state with outer product
                S = alpha_t.unsqueeze(-1) * S + torch.einsum('bhd,bhe->bhde', k_t, v_t)
            
            # Collect chunk outputs
            chunk_out = torch.cat(chunk_output, dim=2).transpose(1, 2).reshape(B, -1, self.dim)
            outputs.append(chunk_out)
        
        return torch.cat(outputs, dim=1)[:, :original_length]