from functools import lru_cache
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


class GemmaRotaryEmbedding(RotaryEmbedding):
    """Half-split RoPE with Hugging Face Gemma's activation-dtype arithmetic."""

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Frequencies are computed in float32 at construction. Gemma rounds
        # cos/sin before multiplying Q/K, which matters for BF16 checkpoints.
        cos_sin = self.cos_sin_cache[positions].to(query.dtype)
        cos, sin = cos_sin.chunk(2, dim=-1)

        def rotate(x):
            first, second = x.chunk(2, dim=-1)
            return torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1)

        return rotate(query), rotate(key)


def get_gemma_rope(head_size: int, max_position: int, base: float) -> GemmaRotaryEmbedding:
    # The model shares its two instances explicitly; a process-global cache
    # could accidentally share mutable buffers across CPU and CUDA models.
    return GemmaRotaryEmbedding(head_size, head_size, max_position, base)


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
):
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
