"""Opt-in-by-hardware end-to-end generation test with no model downloads.

This is skipped on macOS/CPU. On Linux with CUDA, FlashAttention and Triton it
uses the actual scheduler, paged cache, sampler, and CUDA graph replay paths.
"""

import importlib.util
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
import torch.distributed as dist
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import Gemma3ForCausalLM, Gemma3TextConfig, PreTrainedTokenizerFast


CUDA_ENGINE_AVAILABLE = (
    sys.platform == "linux"
    and torch.cuda.is_available()
    and importlib.util.find_spec("flash_attn") is not None
    and importlib.util.find_spec("triton") is not None
)


@unittest.skipUnless(CUDA_ENGINE_AVAILABLE, "requires Linux, an NVIDIA GPU, FlashAttention and Triton")
class CUDAEngineGenerationTests(unittest.TestCase):
    @torch.inference_mode()
    def test_eager_and_graph_generation_match_hf_with_chunking_and_prefix_reuse(self):
        from minivllm import LLM, SamplingParams

        torch.manual_seed(123)
        config = Gemma3TextConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=1,
            head_dim=8,
            query_pre_attn_scalar=8,
            sliding_window=32,
            layer_types=["sliding_attention", "full_attention", "sliding_attention"],
            max_position_embeddings=512,
            attention_dropout=0.0,
            tie_word_embeddings=True,
        )
        config._attn_implementation = "eager"
        reference = Gemma3ForCausalLM(config).eval()
        for name, parameter in reference.named_parameters():
            if "norm" in name:
                parameter.uniform_(-0.1, 0.1)
        reference.to(dtype=torch.float16)

        # The full shared block is reusable; each request has a distinct tail.
        prefix = [2] + [4 + index % 60 for index in range(255)]
        prompts = [prefix + [9], prefix + [10, 11, 12], prefix + [13, 14, 15, 16, 17]]
        max_tokens = 4
        with tempfile.TemporaryDirectory() as directory:
            reference.save_pretrained(directory, safe_serialization=True)
            vocabulary = {"<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3}
            vocabulary.update({f"token{index}": index for index in range(4, 64)})
            backend = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
            backend.pre_tokenizer = Whitespace()
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=backend,
                pad_token="<pad>",
                eos_token="<eos>",
                bos_token="<bos>",
                unk_token="<unk>",
            )
            tokenizer.save_pretrained(directory)

            reference.cuda()
            expected = []
            for prompt in prompts:
                history = prompt.copy()
                for _ in range(max_tokens):
                    result = reference(
                        input_ids=torch.tensor([history], device="cuda"), use_cache=False,
                    )
                    history.append(int(result.logits[0, -1].argmax()))
                expected.append(history[len(prompt):])
            del result, reference
            torch.cuda.empty_cache()

            for eager in (True, False):
                with self.subTest(enforce_eager=eager):
                    calls = []
                    with LLM(
                        directory,
                        enforce_eager=eager,
                        tensor_parallel_size=1,
                        max_model_len=512,
                        max_num_seqs=3,
                        max_num_batched_tokens=64,
                        gpu_memory_utilization=0.8,
                        num_kvcache_blocks=16,
                    ) as engine:
                        original_run = engine.model_runner.run

                        def record_run(sequences, is_prefill):
                            calls.append((is_prefill, len(sequences), sum(s.num_scheduled_tokens for s in sequences)))
                            return original_run(sequences, is_prefill)

                        params = SamplingParams(temperature=0, ignore_eos=True, max_tokens=max_tokens)
                        with patch.object(engine.model_runner, "run", side_effect=record_run):
                            first = engine.generate(prompts, params, use_tqdm=False)
                            manager = engine.scheduler.block_manager
                            with patch.object(manager, "allocate", wraps=manager.allocate) as allocations:
                                second = engine.generate(prompts, params, use_tqdm=False)
                            self.assertTrue(any(call.args[1] >= 1 for call in allocations.call_args_list))
                        self.assertEqual([output["token_ids"] for output in first], expected)
                        self.assertEqual([output["token_ids"] for output in second], expected)
                        self.assertTrue(any(prefill and tokens == 64 for prefill, _, tokens in calls))
                        self.assertTrue(any(not prefill and batch == 3 for prefill, batch, _ in calls))
                        if not eager:
                            self.assertIn(3, engine.model_runner.graphs)
                    self.assertFalse(dist.is_initialized(), "Closing an engine must release its process group")
                    del original_run, engine
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
