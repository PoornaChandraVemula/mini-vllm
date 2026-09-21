import os
from dataclasses import dataclass

import torch
from transformers import AutoConfig, GenerationConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 4096
    max_num_seqs: int = 64
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: object | None = None
    eos: tuple[int, ...] = ()
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        self.model = os.path.abspath(os.path.expanduser(self.model))
        if not os.path.isdir(self.model):
            raise ValueError('model must be a local Hugging Face checkpoint directory')
        if self.tensor_parallel_size != 1:
            raise ValueError('Gemma 3 1B has one KV head; this implementation supports tensor_parallel_size=1')
        if self.kvcache_block_size <= 0 or self.kvcache_block_size % 256:
            raise ValueError('kvcache_block_size must be a positive multiple of 256')
        if min(self.max_num_batched_tokens, self.max_num_seqs, self.max_model_len) <= 0:
            raise ValueError('token, sequence, and context limits must be positive')
        if self.num_kvcache_blocks != -1 and self.num_kvcache_blocks <= 0:
            raise ValueError('num_kvcache_blocks must be -1 (automatic) or positive')
        if not 0 < self.gpu_memory_utilization < 1:
            raise ValueError('gpu_memory_utilization must be between 0 and 1')
        self.hf_config = AutoConfig.from_pretrained(self.model, local_files_only=True)
        if self.hf_config.model_type != 'gemma3_text':
            raise ValueError('Only text-only Gemma 3 checkpoints (1B PT or IT) are supported')
        dtype = self.hf_config.dtype
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype, None)
        if dtype not in (torch.bfloat16, torch.float16):
            raise ValueError('FlashAttention requires checkpoint dtype bfloat16 or float16')
        self.hf_config.dtype = dtype
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        stops = self.hf_config.eos_token_id
        if os.path.isfile(os.path.join(self.model, 'generation_config.json')):
            stops = GenerationConfig.from_pretrained(self.model, local_files_only=True).eos_token_id or stops
        self.eos = tuple(stops) if isinstance(stops, (list, tuple)) else (() if stops is None else (stops,))
