import atexit
from dataclasses import fields
from time import perf_counter
import sys

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from minivllm.config import Config
from minivllm.sampling_params import SamplingParams
from minivllm.engine.sequence import Sequence
from minivllm.engine.scheduler import Scheduler
from minivllm.engine.model_runner import ModelRunner


class LLMEngine:
    def __init__(self, model, **kwargs):
        if sys.platform != 'linux' or not torch.cuda.is_available():
            raise RuntimeError('mini-vllm generation requires Linux and an NVIDIA CUDA GPU; run the CPU tests on macOS')
        allowed = {field.name for field in fields(Config)} - {'model', 'hf_config', 'eos'}
        unknown = kwargs.keys() - allowed
        if unknown:
            raise TypeError(f'Unknown engine options: {sorted(unknown)}')
        config = Config(model, **kwargs)
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True, local_files_only=True)
        stops = set(config.eos)
        if self.tokenizer.eos_token_id is not None:
            stops.add(self.tokenizer.eos_token_id)
        config.eos = tuple(sorted(stops))
        Sequence.block_size = config.kvcache_block_size
        self.model_runner = ModelRunner(config, 0, [])
        self.scheduler = Scheduler(config)
        self._closed = False
        atexit.register(self.exit)

    def exit(self):
        if getattr(self, '_closed', True):
            return
        self._closed = True
        atexit.unregister(self.exit)
        try:
            self.model_runner.call('exit')
        finally:
            del self.model_runner

    close = exit

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.exit()

    def _sequence(self, prompt, sampling_params):
        if self._closed:
            raise RuntimeError('Engine is closed')
        if not isinstance(sampling_params, SamplingParams):
            raise TypeError('sampling_params must be SamplingParams')
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        if not isinstance(prompt, list) or not prompt:
            raise ValueError('Prompt must be text or a nonempty list of token IDs')
        if any(not isinstance(token, int) or isinstance(token, bool) or not 0 <= token < self.config.hf_config.vocab_size for token in prompt):
            raise ValueError('Prompt token IDs must be integers within the model vocabulary')
        if len(prompt) >= self.config.max_model_len:
            raise ValueError(f'Prompt must have fewer than {self.config.max_model_len} tokens to leave room for output')
        return Sequence(prompt, sampling_params)

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        seq = self._sequence(prompt, sampling_params)
        self.scheduler.add(seq)
        return seq.seq_id

    def step(self):
        if self._closed:
            raise RuntimeError('Engine is closed')
        if self.is_finished():
            return [], 0
        seqs, is_prefill = self.scheduler.schedule()
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        token_ids = self.model_runner.call('run', seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(self, prompts: list[str] | list[list[int]],
                 sampling_params: SamplingParams | list[SamplingParams],
                 use_tqdm: bool = True) -> list[dict]:
        if not self.is_finished():
            raise RuntimeError('Finish pending add_request/step work before calling generate')
        params = sampling_params if isinstance(sampling_params, list) else [sampling_params] * len(prompts)
        if len(params) != len(prompts):
            raise ValueError('One SamplingParams is required for each prompt')
        seqs = [self._sequence(prompt, sp) for prompt, sp in zip(prompts, params)]
        for seq in seqs:
            self.scheduler.add(seq)
        outputs = {}
        with tqdm(total=len(prompts), desc='Generating', dynamic_ncols=True, disable=not use_tqdm) as pbar:
            while not self.is_finished():
                start = perf_counter()
                output, num_tokens = self.step()
                phase = 'Prefill' if num_tokens > 0 else 'Decode'
                pbar.set_postfix({phase: f'{int(abs(num_tokens) / max(perf_counter() - start, 1e-9))}tok/s'})
                for seq_id, token_ids in output:
                    outputs[seq_id] = token_ids
                    pbar.update(1)
        return [{'text': self.tokenizer.decode(outputs[seq.seq_id], skip_special_tokens=True),
                 'token_ids': outputs[seq.seq_id]} for seq in seqs]
