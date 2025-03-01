from torch import nn
import torch

class RoPE2D(nn.Module):
    """
    2D Rotary Position Embedding for transformers
    Implements axial rotary embeddings separately for horizontal and vertical directions
    """
    def __init__(self, dim, max_h=256, max_w=256, base=10000.0):
        super().__init__()
        self.dim = dim
        self.max_h = max_h
        self.max_w = max_w
        self.base = base
        
        # Dimension must be even for rotary embeddings
        assert dim % 4 == 0, "Dimension must be divisible by 4 for 2D RoPE"
        
        # Each direction (height and width) gets 1/4 of dimensions
        self.h_dim = dim // 4
        self.w_dim = dim // 4
        
        # Create theta parameters for rotary embeddings
        self.register_buffer(
            "inv_freq_h", 
            1.0 / (base ** (torch.arange(0, self.h_dim, 2).float() / self.h_dim))
        )
        self.register_buffer(
            "inv_freq_w", 
            1.0 / (base ** (torch.arange(0, self.w_dim, 2).float() / self.w_dim))
        )
        
    def _get_rotary_embedding(self, seq_len, inv_freq):
        """Get rotary embedding for a single dimension (height or width)"""
        pos = torch.arange(seq_len, device=inv_freq.device).float()
        freqs = torch.outer(pos, inv_freq)
        
        # Create embedding of shape (seq_len, dim)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Complex rotation using cos/sin
        cos = emb.cos()
        sin = emb.sin()
        
        return cos, sin
    
    def _rotate_half(self, x):
        """Rotate half the hidden dims of x"""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    
    def _apply_rotary_embedding(self, x, cos, sin, dim_idx):
        """Apply rotary embedding along a specific dimension"""
        # Extract the part of x that will be rotated
        # dim_idx: 0 for height, 1 for width
        dim_size = self.h_dim if dim_idx == 0 else self.w_dim
        start_idx = dim_idx * dim_size
        end_idx = start_idx + dim_size
        
        x_to_rotate = x[..., start_idx:end_idx]
        x_rotated = x_to_rotate * cos + self._rotate_half(x_to_rotate) * sin
        
        # Replace the rotated part in x
        x = torch.cat([
            x[..., :start_idx], 
            x_rotated, 
            x[..., end_idx:]
        ], dim=-1)
        
        return x
    
    def forward(self, x):
        """
        Apply 2D rotary embeddings to the input tensor
        Args:
            x: Input tensor of shape [batch, height, width, channels]
        """
        b, h, w, c = x.shape
        
        # Get height and width embeddings
        h_cos, h_sin = self._get_rotary_embedding(h, self.inv_freq_h)  # [h, h_dim//2]
        w_cos, w_sin = self._get_rotary_embedding(w, self.inv_freq_w)  # [w, w_dim//2]
        
        # Apply height rotary embedding (along rows)
        h_cos = h_cos.unsqueeze(1)  # [h, 1, h_dim//2]
        h_sin = h_sin.unsqueeze(1)  # [h, 1, h_dim//2]
        
        x = self._apply_rotary_embedding(x, h_cos, h_sin, 0)
        
        # Apply width rotary embedding (along columns)
        w_cos = w_cos.unsqueeze(0)  # [1, w, w_dim//2]
        w_sin = w_sin.unsqueeze(0)  # [1, w, w_dim//2]
        
        x = self._apply_rotary_embedding(x, w_cos, w_sin, 1)
        
        return x