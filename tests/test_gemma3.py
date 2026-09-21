"""Numerical Gemma 3 checks against a locally initialized HF model.

Run: TORCHDYNAMO_DISABLE=1 python -m unittest discover -s tests -v
No downloaded checkpoint, Hugging Face account, or GPU is needed for CPU tests.
The CUDA suite additionally exercises real production attention/cache kernels.
"""

import importlib.util
import os
import tempfile
import unittest
from contextlib import ExitStack
from unittest.mock import patch

os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
import torch.distributed as dist
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import Gemma3ForCausalLM as HFGemma3ForCausalLM
from transformers import Gemma3TextConfig
from transformers.models.gemma3.modeling_gemma3 import apply_rotary_pos_emb

from minivllm.layers.attention import Attention
from minivllm.models.gemma3 import Gemma3ForCausalLM
from minivllm.utils.context import reset_context, set_context
from minivllm.utils.loader import load_model

try:
    from .reference_attention import reference_attention
except ImportError:  # unittest discovery imports the tests as top-level modules.
    from reference_attention import reference_attention


def tiny_config():
    # hidden_size / heads != head_dim and query scaling != head_dim on purpose.
    config = Gemma3TextConfig(
        vocab_size=64,
        hidden_size=24,
        intermediate_size=48,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        query_pre_attn_scalar=6,
        sliding_window=4,
        layer_types=["sliding_attention", "full_attention", "sliding_attention"],
        rope_theta=1_000_000.0,
        rope_local_base_freq=10_000.0,
        tie_word_embeddings=True,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    return config


class GemmaParityMixin:
    device = "cpu"
    dtype = torch.float32
    block_size = 4
    rtol = 3e-5
    atol = 3e-5
    use_reference_attention = True

    def setUp(self):
        self.contexts = ExitStack()
        self.addCleanup(self.contexts.close)
        self.addCleanup(reset_context)
        self.contexts.enter_context(patch.object(dist, "get_rank", return_value=0))
        self.contexts.enter_context(patch.object(dist, "get_world_size", return_value=1))
        if self.use_reference_attention:
            self.contexts.enter_context(patch.object(Attention, "forward", reference_attention))
        self.directory = self.contexts.enter_context(tempfile.TemporaryDirectory())
        torch.manual_seed(47)
        self.config = tiny_config()
        self.reference = HFGemma3ForCausalLM(self.config).eval()
        with torch.no_grad():
            for name, parameter in self.reference.named_parameters():
                # HF initializes Gemma norm weights to zero. Nonzero values catch
                # accidentally implementing standard RMSNorm instead of (1+w).
                if "norm" in name:
                    parameter.uniform_(-0.2, 0.2)
        self.reference.save_pretrained(self.directory, safe_serialization=True)
        self.model = Gemma3ForCausalLM(self.config).eval()
        load_model(self.model, self.directory)
        self.reference.to(device=self.device, dtype=self.dtype)
        self.model.to(device=self.device, dtype=self.dtype)

    def tensor(self, values, dtype=torch.long):
        return torch.tensor(values, dtype=dtype, device=self.device)

    def assert_parity(self, actual, expected):
        torch.testing.assert_close(actual, expected, rtol=self.rtol, atol=self.atol)

    @torch.inference_mode()
    def reference_result(self, histories, counts):
        hidden, logits = [], []
        for history, count in zip(histories, counts):
            result = self.reference(
                input_ids=self.tensor([history]),
                use_cache=False,
                output_hidden_states=True,
            )
            hidden.append(result.hidden_states[-1][0, -count:])
            logits.append(result.logits[0, -1])
        return torch.cat(hidden), torch.stack(logits)

    def allocate_cache(self):
        # Deliberately disordered physical pages catch accidental contiguous
        # addressing. The two requests never overlap their allocated pages.
        tables = self.tensor([[6, 1, 5, 0], [3, 7, 2, 4]], dtype=torch.int32)
        for module in self.model.modules():
            if isinstance(module, Attention):
                shape = (8, self.block_size, module.num_kv_heads, module.head_dim)
                module.k_cache = torch.zeros(shape, device=self.device, dtype=self.dtype)
                module.v_cache = torch.zeros_like(module.k_cache)
        return tables

    def slots(self, tables, starts, counts):
        return self.tensor([
            int(tables[row, position // self.block_size]) * self.block_size
            + position % self.block_size
            for row, (start, count) in enumerate(zip(starts, counts))
            for position in range(start, start + count)
        ], dtype=torch.int32)

    def prefill_context(self, counts, lengths, tables=None, starts=None):
        def cumulative(values):
            output = [0]
            for value in values:
                output.append(output[-1] + value)
            return self.tensor(output, dtype=torch.int32)

        set_context(
            True,
            cu_seqlens_q=cumulative(counts),
            cu_seqlens_k=cumulative(lengths),
            max_seqlen_q=max(counts),
            max_seqlen_k=max(lengths),
            slot_mapping=None if tables is None else self.slots(tables, starts, counts),
            block_tables=tables,
        )

    @torch.inference_mode()
    def test_packed_prefill_matches_hf_for_mixed_lengths(self):
        histories = [[2, 13, 6, 17, 29, 4, 9, 23, 18, 7, 31], [2, 16, 3, 25, 8, 14, 11]]
        counts = list(map(len, histories))
        self.prefill_context(counts, counts)
        hidden = self.model(
            self.tensor([token for history in histories for token in history]),
            self.tensor([position for count in counts for position in range(count)]),
        )
        logits = self.model.compute_logits(hidden)
        expected_hidden, expected_logits = self.reference_result(histories, counts)
        self.assert_parity(hidden, expected_hidden)
        self.assert_parity(logits, expected_logits)

    @torch.inference_mode()
    def test_paged_chunked_prefill_and_decode_match_hf(self):
        tables = self.allocate_cache()
        histories = [[], []]
        # Both sequences cross the local-attention window and several cache
        # blocks, while each prefill call has different query/key lengths.
        for chunks in (
            [[2, 13, 6, 17, 29, 4], [2, 16, 3]],
            [[9, 23, 18, 7, 31], [25, 8, 14, 11]],
        ):
            starts = list(map(len, histories))
            counts = list(map(len, chunks))
            for history, chunk in zip(histories, chunks):
                history.extend(chunk)
            lengths = list(map(len, histories))
            self.prefill_context(counts, lengths, tables, starts)
            hidden = self.model(
                self.tensor([token for chunk in chunks for token in chunk]),
                self.tensor([
                    position
                    for start, count in zip(starts, counts)
                    for position in range(start, start + count)
                ]),
            )
            expected_hidden, expected_logits = self.reference_result(histories, counts)
            self.assert_parity(hidden, expected_hidden)
            self.assert_parity(self.model.compute_logits(hidden), expected_logits)

        for tokens in ([22, 45], [38, 12]):
            starts = list(map(len, histories))
            for history, token in zip(histories, tokens):
                history.append(token)
            set_context(
                False,
                slot_mapping=self.slots(tables, starts, [1, 1]),
                context_lens=self.tensor(list(map(len, histories)), dtype=torch.int32),
                block_tables=tables,
            )
            hidden = self.model(self.tensor(tokens), self.tensor(starts))
            expected_hidden, expected_logits = self.reference_result(histories, [1, 1])
            self.assert_parity(hidden, expected_hidden)
            self.assert_parity(self.model.compute_logits(hidden), expected_logits)

    def test_tied_checkpoint_and_packed_weights_are_loaded(self):
        with safe_open(os.path.join(self.directory, "model.safetensors"), framework="pt") as checkpoint:
            self.assertNotIn("lm_head.weight", checkpoint.keys())
        self.assertEqual(self.model.lm_head.weight.data_ptr(), self.model.model.embed_tokens.weight.data_ptr())
        self.assert_parity(self.model.model.embed_tokens.weight, self.reference.model.embed_tokens.weight)
        for actual_layer, expected_layer in zip(self.model.model.layers, self.reference.model.layers):
            expected_qkv = torch.cat([
                expected_layer.self_attn.q_proj.weight,
                expected_layer.self_attn.k_proj.weight,
                expected_layer.self_attn.v_proj.weight,
            ])
            expected_gate_up = torch.cat([expected_layer.mlp.gate_proj.weight, expected_layer.mlp.up_proj.weight])
            self.assert_parity(actual_layer.self_attn.qkv_proj.weight, expected_qkv)
            self.assert_parity(actual_layer.mlp.gate_up_proj.weight, expected_gate_up)


class Gemma3CPUParityTests(GemmaParityMixin, unittest.TestCase):
    """Model math/weights oracle, with GPU attention deliberately substituted."""

    def checkpoint_tensors(self):
        with safe_open(os.path.join(self.directory, "model.safetensors"), framework="pt") as checkpoint:
            return {name: checkpoint.get_tensor(name) for name in checkpoint.keys()}

    def test_loader_rejects_directory_without_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "No .safetensors weights"):
                load_model(self.model, directory)

    def test_loader_rejects_missing_fused_projection_shard(self):
        tensors = self.checkpoint_tensors()
        del tensors["model.layers.0.self_attn.k_proj.weight"]
        with tempfile.TemporaryDirectory() as directory:
            save_file(tensors, os.path.join(directory, "model.safetensors"))
            with self.assertRaisesRegex(ValueError, r"missing required weights.*qkv_proj.weight\[k\]"):
                load_model(self.model, directory)

    def test_loader_rejects_unexpected_checkpoint_tensor(self):
        tensors = self.checkpoint_tensors()
        tensors["model.unknown.weight"] = torch.ones(1)
        with tempfile.TemporaryDirectory() as directory:
            save_file(tensors, os.path.join(directory, "model.safetensors"))
            with self.assertRaisesRegex(ValueError, r"Unexpected checkpoint tensor: model.unknown.weight"):
                load_model(self.model, directory)

    def test_loader_rejects_broadcastable_packed_projection_shape(self):
        tensors = self.checkpoint_tensors()
        name = "model.layers.0.self_attn.q_proj.weight"
        # Tensor.copy_ can silently broadcast a single row to every query head.
        # A complete checkpoint must be validated before invoking that copy.
        tensors[name] = tensors[name][:1].contiguous()
        with tempfile.TemporaryDirectory() as directory:
            save_file(tensors, os.path.join(directory, "model.safetensors"))
            with self.assertRaisesRegex(ValueError, "shape"):
                load_model(self.model, directory)

    @torch.inference_mode()
    def test_reduced_precision_norm_and_rope_match_hf(self):
        # Full FP32 parity cannot catch casting before (1+w), or failing to
        # round RoPE cos/sin before the activation-dtype multiplication.
        positions = torch.arange(61)
        for dtype in (torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                hidden = torch.randn(61, self.config.hidden_size).to(dtype)
                self.model.model.norm.to(dtype)
                self.reference.model.norm.to(dtype)
                torch.testing.assert_close(
                    self.model.model.norm(hidden), self.reference.model.norm(hidden),
                    rtol=0, atol=0,
                )
                query = torch.randn(61, self.config.num_attention_heads, self.config.head_dim).to(dtype)
                key = torch.randn(61, self.config.num_key_value_heads, self.config.head_dim).to(dtype)
                for layer_index, reference_rope in (
                    (0, self.reference.model.rotary_emb_local),
                    (1, self.reference.model.rotary_emb),
                ):
                    actual = self.model.model.layers[layer_index].self_attn.rotary_emb(positions, query, key)
                    cos, sin = reference_rope(hidden.unsqueeze(0), positions.unsqueeze(0))
                    expected = apply_rotary_pos_emb(
                        query.unsqueeze(0), key.unsqueeze(0), cos, sin, unsqueeze_dim=2,
                    )
                    for actual_value, expected_value in zip(actual, expected):
                        torch.testing.assert_close(actual_value, expected_value[0], rtol=0, atol=0)


@unittest.skipUnless(
    torch.cuda.is_available() and importlib.util.find_spec("flash_attn") is not None,
    "requires a CUDA GPU and FlashAttention; CPU tests do not validate CUDA kernels",
)
class Gemma3CUDAParityTests(GemmaParityMixin, unittest.TestCase):
    """Real FlashAttention/Triton forward, paged prefill, and cached decode."""

    device = "cuda"
    dtype = torch.float16
    # FlashAttention's paged KV interface requires a multiple of 256.
    block_size = 256
    rtol = 2e-2
    atol = 2e-2
    use_reference_attention = False


if __name__ == "__main__":
    unittest.main()
