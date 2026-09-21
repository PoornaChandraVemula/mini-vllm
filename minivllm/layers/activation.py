import torch
from torch import nn
import torch.nn.functional as F


class GeluAndMul(nn.Module):
    """Gemma's gated MLP uses tanh-approximate GELU on the gate half."""

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = x.chunk(2, -1)
        return F.gelu(gate, approximate="tanh") * up


class SiluAndMul(nn.Module):

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y
