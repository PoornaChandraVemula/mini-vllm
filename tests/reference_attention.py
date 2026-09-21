"""Small, independent attention oracle used only by the CPU model tests.

This deliberately uses ordinary PyTorch operations instead of the production
FlashAttention/Triton kernels. It is not an alternative inference backend.
"""

import torch

from minivllm.utils.context import get_context


def reference_attention(self, query, key, value):
    context = get_context()
    has_cache = self.k_cache.numel() > 0
    if has_cache:
        slots = context.slot_mapping.long()
        valid = slots >= 0
        self.k_cache.flatten(0, 1)[slots[valid]] = key[valid]
        self.v_cache.flatten(0, 1)[slots[valid]] = value[valid]

    if context.is_prefill:
        query_bounds = context.cu_seqlens_q.tolist()
        key_bounds = context.cu_seqlens_k.tolist()
        lengths = [b - a for a, b in zip(key_bounds, key_bounds[1:])]
    else:
        query_bounds = list(range(query.shape[0] + 1))
        lengths = context.context_lens.tolist()

    outputs = []
    for index, (begin, end) in enumerate(zip(query_bounds, query_bounds[1:])):
        q = query[begin:end]
        length = lengths[index]
        if context.block_tables is not None:
            assert has_cache, "paged attention requires allocated KV storage"
            block_size = self.k_cache.shape[1]
            num_blocks = (length + block_size - 1) // block_size
            pages = context.block_tables[index, :num_blocks].long()
            k = self.k_cache[pages].flatten(0, 1)[:length]
            v = self.v_cache[pages].flatten(0, 1)[:length]
        else:
            assert context.is_prefill
            k = key[key_bounds[index]:key_bounds[index + 1]]
            v = value[key_bounds[index]:key_bounds[index + 1]]

        # A KV head is shared by a contiguous group of query heads.
        repeats = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)
        scores = torch.einsum("qhd,khd->hqk", q, k) * self.scale

        # Cached chunks use bottom-right causal alignment: the chunk is the
        # newest suffix of a longer key sequence, not a fresh sequence at zero.
        query_positions = torch.arange(length - len(q), length, device=q.device)
        key_positions = torch.arange(length, device=q.device)
        allowed = key_positions[None, :] <= query_positions[:, None]
        left, right = self.window_size
        if left >= 0:
            allowed &= key_positions[None, :] >= query_positions[:, None] - left
        if right >= 0:
            allowed &= key_positions[None, :] <= query_positions[:, None] + right
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        probabilities = scores.softmax(dim=-1, dtype=torch.float32).to(q.dtype)
        outputs.append(torch.einsum("hqk,khd->qhd", probabilities, v))

    result = torch.cat(outputs)
    return result if context.is_prefill else result.unsqueeze(1)
