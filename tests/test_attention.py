"""CPU checks of the production GPU-call contract, without GPU kernels.

These spies validate forwarding of windows, lengths, cache tensors and slots.
Numerical attention correctness is covered separately by the parity suites.
"""

import os
import sys
import types
import unittest
from unittest.mock import Mock, patch

os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch

from minivllm.layers.attention import Attention
from minivllm.layers.sampler import Sampler
from minivllm.utils.context import reset_context, set_context


class AttentionKernelContractTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(reset_context)
        self.result = torch.ones(1)
        self.calls = Mock()
        self.prefill = Mock(return_value=self.result)
        self.decode = Mock(return_value=self.result)
        self.store = Mock()
        self.calls.attach_mock(self.prefill, "prefill")
        self.calls.attach_mock(self.decode, "decode")
        self.calls.attach_mock(self.store, "store")
        flash = types.ModuleType("flash_attn")
        flash.flash_attn_varlen_func = self.prefill
        flash.flash_attn_with_kvcache = self.decode
        cache = types.ModuleType("minivllm.layers.kvcache")
        cache.store_kvcache = self.store
        modules = patch.dict(sys.modules, {"flash_attn": flash, "minivllm.layers.kvcache": cache})
        modules.start()
        self.addCleanup(modules.stop)

    def tensors(self, token_count):
        return (
            torch.randn(token_count, 2, 8),
            torch.randn(token_count, 1, 8),
            torch.randn(token_count, 1, 8),
        )

    def attention(self, window, cached=False):
        attention = Attention(2, 8, 0.125, 1, window_size=window)
        if cached:
            attention.k_cache = torch.randn(4, 4, 1, 8)
            attention.v_cache = torch.randn_like(attention.k_cache)
        return attention

    def assert_arguments_are(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for actual_value, expected_value in zip(actual, expected):
            self.assertIs(actual_value, expected_value)

    def assert_common_options(self, options, window):
        self.assertEqual(options["window_size"], window)
        self.assertEqual(options["softmax_scale"], 0.125)
        self.assertIs(options["causal"], True)

    def test_packed_prefill_forwards_local_and_global_windows(self):
        for window in ((4 - 1, 0), (-1, -1)):
            with self.subTest(window=window):
                self.calls.reset_mock()
                attention = self.attention(window)
                query, key, value = self.tensors(5)
                cumulative = torch.tensor([0, 2, 5], dtype=torch.int32)
                set_context(
                    True, cu_seqlens_q=cumulative, cu_seqlens_k=cumulative,
                    max_seqlen_q=3, max_seqlen_k=3,
                )
                self.assertIs(attention(query, key, value), self.result)
                self.prefill.assert_called_once()
                self.store.assert_not_called()
                self.decode.assert_not_called()
                arguments, options = self.prefill.call_args
                self.assert_arguments_are(arguments, (query, key, value))
                self.assert_common_options(options, window)
                self.assertIs(options["cu_seqlens_q"], cumulative)
                self.assertIs(options["cu_seqlens_k"], cumulative)
                self.assertEqual(options["max_seqlen_q"], 3)
                self.assertEqual(options["max_seqlen_k"], 3)
                self.assertIsNone(options["block_table"])

    def test_cached_chunked_prefill_passes_paged_cache_and_full_key_lengths(self):
        for window in ((4 - 1, 0), (-1, -1)):
            with self.subTest(window=window):
                self.calls.reset_mock()
                attention = self.attention(window, cached=True)
                query, key, value = self.tensors(5)
                query_lengths = torch.tensor([0, 2, 5], dtype=torch.int32)
                key_lengths = torch.tensor([0, 7, 11], dtype=torch.int32)
                tables = torch.tensor([[1, 0], [2, 3]], dtype=torch.int32)
                slots = torch.tensor([1, 2, 9, 10, 11], dtype=torch.int32)
                set_context(
                    True, cu_seqlens_q=query_lengths, cu_seqlens_k=key_lengths,
                    max_seqlen_q=3, max_seqlen_k=7,
                    slot_mapping=slots, block_tables=tables,
                )
                self.assertIs(attention(query, key, value), self.result)
                self.assertEqual([call[0] for call in self.calls.mock_calls], ["store", "prefill"])
                self.assert_arguments_are(
                    self.store.call_args.args,
                    (key, value, attention.k_cache, attention.v_cache, slots),
                )
                arguments, options = self.prefill.call_args
                self.assert_arguments_are(arguments, (query, attention.k_cache, attention.v_cache))
                self.assert_common_options(options, window)
                self.assertIs(options["cu_seqlens_q"], query_lengths)
                self.assertIs(options["cu_seqlens_k"], key_lengths)
                self.assertIs(options["block_table"], tables)
                self.assertEqual(options["max_seqlen_q"], 3)
                self.assertEqual(options["max_seqlen_k"], 7)

    def test_paged_decode_stores_slots_and_forwards_local_and_global_windows(self):
        for window in ((4 - 1, 0), (-1, -1)):
            with self.subTest(window=window):
                self.calls.reset_mock()
                attention = self.attention(window, cached=True)
                query, key, value = self.tensors(2)
                tables = torch.tensor([[1, 0], [2, 3]], dtype=torch.int32)
                lengths = torch.tensor([7, 4], dtype=torch.int32)
                slots = torch.tensor([2, 11], dtype=torch.int32)
                set_context(False, slot_mapping=slots, context_lens=lengths, block_tables=tables)
                self.assertIs(attention(query, key, value), self.result)
                self.assertEqual([call[0] for call in self.calls.mock_calls], ["store", "decode"])
                self.assert_arguments_are(
                    self.store.call_args.args,
                    (key, value, attention.k_cache, attention.v_cache, slots),
                )
                arguments, options = self.decode.call_args
                torch.testing.assert_close(arguments[0], query.unsqueeze(1), rtol=0, atol=0)
                self.assert_arguments_are(arguments[1:], (attention.k_cache, attention.v_cache))
                self.assert_common_options(options, window)
                self.assertIs(options["cache_seqlens"], lengths)
                self.assertIs(options["block_table"], tables)


class SamplerTests(unittest.TestCase):
    def test_mixed_greedy_and_sampling_preserves_inputs(self):
        torch.manual_seed(29)
        logits = torch.tensor([
            [-4.0, -1.0, 8.0, 0.0, 2.0],
            [0.0, 2.0, -1.0, 1.0, 0.0],
            [4.0, 4.0, 1.0, 0.0, -2.0],
            [-3.0, 2.0, 4.0, 1.0, 0.0],
        ])
        temperatures = torch.tensor([0.0, 0.5, 0.0, 1.5])
        original_logits, original_temperatures = logits.clone(), temperatures.clone()
        tokens = Sampler()(logits, temperatures)
        self.assertEqual(tuple(tokens.shape), (4,))
        self.assertEqual(tokens.dtype, torch.int64)
        self.assertEqual(tokens[[0, 2]].tolist(), logits[[0, 2]].argmax(dim=-1).tolist())
        self.assertTrue(bool(((tokens >= 0) & (tokens < logits.shape[1])).all()))
        torch.testing.assert_close(logits, original_logits, rtol=0, atol=0)
        torch.testing.assert_close(temperatures, original_temperatures, rtol=0, atol=0)

    def test_positive_temperature_samples_multiple_vocabulary_entries(self):
        torch.manual_seed(73)
        tokens = Sampler()(torch.zeros(256, 8), torch.ones(256))
        self.assertTrue(bool(((tokens >= 0) & (tokens < 8)).all()))
        self.assertGreater(tokens.unique().numel(), 1, "Positive temperature must not always choose argmax")


if __name__ == "__main__":
    unittest.main()
