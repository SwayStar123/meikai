import torch
import torch.nn as nn
import torch.nn.functional as F
from natten import NeighborhoodAttention2D
from .embed import RoPE2D

class GEGLU(nn.Module):
    """Gate-Enhanced Gated Linear Unit module with zero-init for the output projection."""
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.ff1 = nn.Linear(dim, dim * 2)
        self.ff2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        # Zero-init output projection
        nn.init.zeros_(self.ff2.weight)
        nn.init.zeros_(self.ff2.bias)
    
    def forward(self, x):
        x = self.ff1(x)
        gate, value = x.chunk(2, dim=-1)
        x = F.gelu(gate) * value
        x = self.dropout(x)
        x = self.ff2(x)
        return x


class NeighborhoodAttentionBlock(nn.Module):
    def __init__(self, dim, head_dim=64, kernel_size=7, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.heads = dim // head_dim
        self.head_dim = head_dim
        
        self.norm1 = nn.RMSNorm([dim])
        
        self.na = NeighborhoodAttention2D(
            dim=dim,
            num_heads=self.heads,
            kernel_size=kernel_size,
            qkv_bias=False
        )
        
        self.norm2 = nn.RMSNorm([dim])
        
        # RoPE positional embedding
        self.pos_embed = RoPE2D(dim)
        
        # GEGLU feedforward
        self.geglu = GEGLU(dim, dropout)
        

    def forward(self, x):
        # x shape: [batch, height, width, channels]
        b, h, w, c = x.shape
        
        # Self-attention with pre-norm
        residual = x
        x = self.norm1(x)
        
        # Apply RoPE positional embedding
        x = self.pos_embed(x)
            
        # NATTEN expects [batch, height, width, channels]
        x_attn = self.na(x)
        x = residual + x_attn
        
        # Feedforward with pre-norm
        residual = x
        x = self.norm2(x)
            
        # GEGLU feedforward
        x_ff = self.geglu(x)
        
        x = residual + x_ff
        return x


class GlobalAttentionBlock(nn.Module):
    def __init__(self, dim, head_dim=64, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.heads = dim // head_dim
        self.head_dim = head_dim
        
        self.norm1 = nn.RMSNorm([dim])
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
        self.norm2 = nn.RMSNorm([dim])
        
        # RoPE positional embedding
        self.pos_embed = RoPE2D(dim)
        
        # GEGLU feedforward
        self.geglu = GEGLU(dim, dropout)
        
        # Zero-init output projection for attention
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        
    def forward(self, x):
        # x shape: [batch, height, width, channels]
        b, h, w, c = x.shape
        n = h * w
        
        # Attention with pre-norm
        residual = x
        x = self.norm1(x)
        
        # Apply RoPE positional embedding
        x = self.pos_embed(x)
        
        # Reshape to sequence
        x_seq = x.reshape(b, n, c)
        
        # Self-attention
        qkv = self.qkv(x_seq).reshape(b, n, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [b, heads, n, head_dim]
        
        # Use PyTorch's built-in scaled dot-product attention
        x_attn = F.scaled_dot_product_attention(q, k, v)
        x_attn = x_attn.transpose(1, 2).reshape(b, n, c)
        x_attn = self.proj(x_attn)
        x_attn = x_attn.reshape(b, h, w, c)
        
        x = residual + x_attn
        
        # Feedforward with pre-norm
        residual = x
        x = self.norm2(x)
        
        # GEGLU feedforward
        x_ff = self.geglu(x)
        
        x = residual + x_ff
        return x
