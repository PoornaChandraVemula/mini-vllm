"""Text-only Gemma 3 with packed projections and paged attention.

The default supported checkpoint is google/gemma-3-1b-pt (or its -it variant).
The model computes the network itself; Transformers only provides configuration
and tokenization. Weight names follow the Hugging Face checkpoint layout.
"""

import torch
from torch import nn
import torch.distributed as dist
from transformers import Gemma3TextConfig

from minivllm.layers.activation import GeluAndMul
from minivllm.layers.attention import Attention
from minivllm.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from minivllm.layers.layernorm import GemmaRMSNorm
from minivllm.layers.linear import MergedColumnParallelLinear, QKVParallelLinear, RowParallelLinear
from minivllm.layers.rotary_embedding import GemmaRotaryEmbedding, get_gemma_rope


def validate_gemma_config(config: Gemma3TextConfig) -> None:
    """Reject architectures whose semantics this small implementation omits."""
    if getattr(config, "model_type", None) != "gemma3_text":
        raise ValueError("mini-vllm supports text-only Gemma 3 checkpoints (model_type='gemma3_text')")
    if config.hidden_activation != "gelu_pytorch_tanh":
        raise ValueError("Gemma 3 requires hidden_activation='gelu_pytorch_tanh'")
    rope_scaling = getattr(config, "rope_scaling", None)
    if rope_scaling and rope_scaling.get("rope_type", rope_scaling.get("type")) != "default":
        raise ValueError("Scaled RoPE is not supported; use the Gemma 3 1B checkpoint")
    if getattr(config, "use_bidirectional_attention", False):
        raise ValueError("Only causal text attention is supported")
    if config.attn_logit_softcapping is not None or config.final_logit_softcapping is not None:
        raise ValueError("Attention/logit softcapping is not supported; use Gemma 3 1B")
    if config.num_key_value_heads <= 0 or config.num_attention_heads % config.num_key_value_heads:
        raise ValueError("Query heads must be divisible by the positive KV head count")
    if config.head_dim <= 0 or config.head_dim % 2 or config.head_dim > 256:
        raise ValueError("FlashAttention requires an even head_dim between 2 and 256")
    if config.query_pre_attn_scalar <= 0 or config.sliding_window <= 0:
        raise ValueError("Attention scale and sliding window must be positive")
    if len(config.layer_types) != config.num_hidden_layers or any(
        layer_type not in {"sliding_attention", "full_attention"} for layer_type in config.layer_types
    ):
        raise ValueError("layer_types must specify sliding_attention or full_attention for each layer")


class Gemma3Attention(nn.Module):
    def __init__(
        self,
        config: Gemma3TextConfig,
        layer_idx: int,
        rotary_emb: GemmaRotaryEmbedding | None = None,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        if config.num_attention_heads % tp_size or config.num_key_value_heads % tp_size:
            raise ValueError("Attention heads and KV heads must divide the tensor parallel size; Gemma 3 1B uses TP=1")
        self.num_heads = config.num_attention_heads // tp_size
        self.num_kv_heads = config.num_key_value_heads // tp_size
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = config.query_pre_attn_scalar ** -0.5
        self.is_sliding = config.layer_types[layer_idx] == "sliding_attention"
        self.qkv_proj = QKVParallelLinear(
            config.hidden_size, self.head_dim, config.num_attention_heads,
            config.num_key_value_heads, bias=config.attention_bias,
        )
        self.o_proj = RowParallelLinear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias,
        )
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        theta = config.rope_local_base_freq if self.is_sliding else config.rope_theta
        self.rotary_emb = rotary_emb if rotary_emb is not None else get_gemma_rope(
            self.head_dim, config.max_position_embeddings, theta,
        )
        # FlashAttention uses inclusive bounds. A 512-token Gemma window
        # includes the query itself and at most 511 preceding tokens.
        self.attn = Attention(
            self.num_heads, self.head_dim, self.scaling, self.num_kv_heads,
            window_size=(config.sliding_window - 1, 0) if self.is_sliding else (-1, -1),
        )

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        q, k, v = self.qkv_proj(hidden_states).split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = self.q_norm(q.reshape(-1, self.num_heads, self.head_dim))
        k = self.k_norm(k.reshape(-1, self.num_kv_heads, self.head_dim))
        v = v.reshape(-1, self.num_kv_heads, self.head_dim)
        q, k = self.rotary_emb(positions, q, k)
        output = self.attn(q, k, v)
        return self.o_proj(output.reshape(-1, self.q_size))


class Gemma3MLP(nn.Module):
    def __init__(self, config: Gemma3TextConfig) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size, [config.intermediate_size] * 2, bias=False,
        )
        self.act_fn = GeluAndMul()
        self.down_proj = RowParallelLinear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_up_proj(x)))


class Gemma3DecoderLayer(nn.Module):
    def __init__(
        self, config: Gemma3TextConfig, layer_idx: int, rotary_emb: GemmaRotaryEmbedding | None = None,
    ) -> None:
        super().__init__()
        self.self_attn = Gemma3Attention(config, layer_idx, rotary_emb)
        self.mlp = Gemma3MLP(config)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.pre_feedforward_layernorm = GemmaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_feedforward_layernorm = GemmaRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        attention = self.self_attn(positions, self.input_layernorm(hidden_states))
        hidden_states = residual + self.post_attention_layernorm(attention)
        residual = hidden_states
        feedforward = self.mlp(self.pre_feedforward_layernorm(hidden_states))
        return residual + self.post_feedforward_layernorm(feedforward)


class Gemma3Model(nn.Module):
    def __init__(self, config: Gemma3TextConfig) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.register_buffer("embed_scale", torch.tensor(config.hidden_size ** 0.5), persistent=False)
        # Share the two position tables within this model. Keep local and
        # global bases separate without repeating tables for all 26 layers.
        rotary_embeddings = {
            "sliding_attention": get_gemma_rope(config.head_dim, config.max_position_embeddings, config.rope_local_base_freq),
            "full_attention": get_gemma_rope(config.head_dim, config.max_position_embeddings, config.rope_theta),
        }
        self.layers = nn.ModuleList([
            Gemma3DecoderLayer(config, i, rotary_embeddings[config.layer_types[i]])
            for i in range(config.num_hidden_layers)
        ])
        self.norm = GemmaRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = hidden_states * self.embed_scale.to(hidden_states.dtype)
        for layer in self.layers:
            hidden_states = layer(positions, hidden_states)
        return self.norm(hidden_states)


class Gemma3ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config: Gemma3TextConfig) -> None:
        super().__init__()
        validate_gemma_config(config)
        self.config = config
        self.model = Gemma3Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)
