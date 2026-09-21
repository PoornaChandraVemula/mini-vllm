import torch
from torch import nn


class Sampler(nn.Module):

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        greedy = logits.argmax(dim=-1)
        scaled = logits.float() / temperatures.clamp_min(1e-10).unsqueeze(1)
        probs = torch.softmax(scaled, dim=-1)
        sampled = (probs / torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return torch.where(temperatures == 0, greedy, sampled)
