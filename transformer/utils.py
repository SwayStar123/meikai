from torch import nn
import torch
import torch.nn.functional as F

class PixelUnshuffle(nn.Module):
    def __init__(self, downscale_factor):
        super().__init__()
        self.downscale_factor = downscale_factor
        
    def forward(self, x):
        # x: [b, h, w, c]
        b, h, w, c = x.shape
        f = self.downscale_factor
        # Return: [b, h/f, w/f, c*f*f]
        return x.reshape(b, h // f, f, w // f, f, c).permute(0, 1, 3, 5, 2, 4).reshape(b, h // f, w // f, c * f * f)

class PixelShuffle(nn.Module):
    def __init__(self, upscale_factor):
        super().__init__()
        self.upscale_factor = upscale_factor
        
    def forward(self, x):
        # x: [b, h, w, c]
        b, h, w, c = x.shape
        f = self.upscale_factor
        # Return: [b, h*f, w*f, c/(f*f)]
        return x.reshape(b, h, w, c // (f * f), f, f).permute(0, 1, 4, 2, 5, 3).reshape(b, h * f, w * f, c // (f * f))

class DownsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, shortcut: bool = True, factor: int = 2) -> None:
        super().__init__()

        self.shortcut = shortcut
        self.factor = factor
        self.stride = factor
        self.group_size = in_channels * self.factor**2 // out_channels
        self.unshuffle = PixelUnshuffle(self.factor)

        self.proj = nn.Linear(in_channels * self.factor**2, out_channels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.proj(self.unshuffle(hidden_states))

        if self.shortcut:
            y = self.unshuffle(hidden_states)
            y = y.unflatten(3, (-1, self.group_size))
            y = y.mean(dim=4)
            hidden_states = x + y
        else:
            hidden_states = x

        return hidden_states

class UpsampleBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        interpolation_mode: str = "nearest",
        factor: int = 2,
        shortcut: bool = True
    ) -> None:
        super().__init__()

        self.shortcut = shortcut
        self.interpolation_mode = interpolation_mode
        self.factor = factor
        self.repeats = out_channels * self.factor**2 // in_channels
        self.shuffle = PixelShuffle(self.factor)
        self.proj = nn.Linear(in_channels // self.factor**2, out_channels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.proj(self.shuffle(hidden_states))

        if self.shortcut:
            y = hidden_states.repeat_interleave(self.repeats, dim=3)
            y = self.shuffle(y)
            hidden_states = x + y
        else:
            hidden_states = x

        return hidden_states
