from natten.functional import na2d
import torch

# Given Q, K and V;
# where q/k/v.shape is [batch, height, width, heads, head_dim]
# NOTE: layout is different from unfused;
# it's batch, spatial extent, then heads, then head_dim.

# Self attn: output = sdpa(q, k, v, scale=attn_scale)
q = torch.randn(1, 10, 10, 16, 16).to("cuda")
k = torch.randn(1, 10, 10, 16, 16).to("cuda")
v = torch.randn(1, 10, 10, 16, 16).to("cuda")

kernel_size = 3
dilation = 1

output = na2d(q, k, v, kernel_size=kernel_size, dilation=dilation)

print(output.shape)
