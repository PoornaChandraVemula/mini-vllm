"""Public input/configuration/lifecycle tests without weights or a CUDA device."""

import os
import tempfile
import unittest
from contextlib import ExitStack
from unittest.mock import patch

os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
from transformers import Gemma3TextConfig, GenerationConfig, LlamaConfig

from minivllm.config import Config
from minivllm.engine.llm_engine import LLMEngine
from minivllm.sampling_params import SamplingParams


def write_config(directory, dtype=torch.bfloat16):
    config = Gemma3TextConfig(
        vocab_size=128,
        hidden_size=24,
        intermediate_size=48,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
        eos_token_id=1,
        dtype=dtype,
    )
    config.save_pretrained(directory)
    return config


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        write_config(self.directory.name)

    def test_generation_config_preserves_all_stop_tokens(self):
        GenerationConfig(eos_token_id=[1, 107]).save_pretrained(self.directory.name)
        config = Config(self.directory.name)
        self.assertEqual(set(config.eos), {1, 107})
        self.assertEqual(config.max_model_len, 32)
        self.assertEqual(config.hf_config.dtype, torch.bfloat16)

    def test_model_eos_is_used_without_generation_config(self):
        self.assertEqual(Config(self.directory.name).eos, (1,))

    def test_unsupported_tensor_parallelism_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "tensor_parallel_size=1"):
            Config(self.directory.name, tensor_parallel_size=2)

    def test_unsupported_model_architecture_is_rejected(self):
        LlamaConfig().save_pretrained(self.directory.name)
        with self.assertRaisesRegex(ValueError, "Gemma 3"):
            Config(self.directory.name)

    def test_float32_checkpoint_is_rejected_before_gpu_allocation(self):
        write_config(self.directory.name, dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "bfloat16 or float16"):
            Config(self.directory.name)


class FakeTokenizer:
    eos_token_id = 1
    vocab_size = 128

    def __len__(self):
        return self.vocab_size

    def encode(self, text, **kwargs):
        return [2, 3] if text else []

    def decode(self, token_ids, **kwargs):
        return ",".join(map(str, token_ids))


class FakeModelRunner:
    def __init__(self, config, *args, **kwargs):
        config.num_kvcache_blocks = 8
        self.exit_count = 0
        self.run_count = 0

    def call(self, method, *args):
        if method == "exit":
            self.exit()
        elif method == "run":
            self.run_count += 1
            seqs, _ = args
            return [seq.prompt_token_ids[0] + 10 for seq in seqs]
        else:
            raise AssertionError(f"Unexpected fake runner method {method}")

    def exit(self):
        self.exit_count += 1


class PublicAPITests(unittest.TestCase):
    def setUp(self):
        self.contexts = ExitStack()
        self.addCleanup(self.contexts.close)
        directory = self.contexts.enter_context(tempfile.TemporaryDirectory())
        write_config(directory)
        GenerationConfig(eos_token_id=[1, 107]).save_pretrained(directory)
        self.contexts.enter_context(patch("minivllm.engine.llm_engine.ModelRunner", FakeModelRunner))
        self.contexts.enter_context(patch(
            "minivllm.engine.llm_engine.AutoTokenizer.from_pretrained",
            return_value=FakeTokenizer(),
        ))
        self.contexts.enter_context(patch("minivllm.engine.llm_engine.atexit.register"))
        self.contexts.enter_context(patch("minivllm.engine.llm_engine.atexit.unregister"))
        # Only the constructor's platform check is bypassed; FakeModelRunner
        # prevents all GPU work and the real scheduler still handles requests.
        with patch("minivllm.engine.llm_engine.sys.platform", "linux"), patch(
            "minivllm.engine.llm_engine.torch.cuda.is_available", return_value=True,
        ):
            self.engine = LLMEngine(directory, max_model_len=16, max_num_batched_tokens=2)
        self.runner = self.engine.model_runner
        self.addCleanup(self.engine.close)

    def test_engine_keeps_instruction_end_of_turn_stop_token(self):
        self.assertEqual(self.engine.scheduler.eos_token_ids, {1, 107})

    def test_mismatched_parameter_list_admits_no_requests(self):
        with self.assertRaises(ValueError):
            self.engine.generate([[2], [3]], [SamplingParams(max_tokens=1)], use_tqdm=False)
        self.assertTrue(self.engine.scheduler.is_finished())
        self.assertEqual(self.runner.run_count, 0)

    def test_invalid_later_prompt_admits_no_earlier_requests(self):
        for prompts in ([[2], []], [[2], [128]], [[2], [2] * 16]):
            with self.subTest(prompts=prompts):
                with self.assertRaises(ValueError):
                    self.engine.generate(prompts, SamplingParams(max_tokens=1), use_tqdm=False)
                self.assertTrue(self.engine.scheduler.is_finished())
                self.assertEqual(self.runner.run_count, 0)

    def test_generation_results_follow_input_order(self):
        outputs = self.engine.generate(
            [[3, 4, 5, 6], [8]], SamplingParams(max_tokens=1), use_tqdm=False,
        )
        self.assertEqual([output["token_ids"] for output in outputs], [[13], [18]])
        self.assertEqual([output["text"] for output in outputs], ["13", "18"])
        self.assertTrue(self.engine.scheduler.is_finished())

    def test_empty_batch_returns_empty_results(self):
        self.assertEqual(self.engine.generate([], SamplingParams(), use_tqdm=False), [])
        self.assertEqual(self.runner.run_count, 0)

    def test_close_is_idempotent(self):
        self.engine.close()
        self.engine.close()
        self.assertEqual(self.runner.exit_count, 1)


if __name__ == "__main__":
    unittest.main()
