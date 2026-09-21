# mini-vllm

An educational reimplementation of core vLLM-style inference techniques for
**Gemma 3 1B**. The project explores paged KV caching, prefix reuse, continuous
batching, chunked prefill, FlashAttention, Triton cache writes, and CUDA graphs
in a compact Python engine.

Built to make transformer inference internals easier to study, experiment with,
and understand.

This is an offline Python inference engine, not an HTTP serving platform.
The custom model computes Gemma directly; Hugging Face supplies configuration,
tokenization, checkpoint files, and the independent test reference.

## Supported setup

- Text-only `google/gemma-3-1b-pt` (completion) and `google/gemma-3-1b-it` (chat).
- Linux, Python 3.10–3.12, an NVIDIA Ampere/Ada/Hopper GPU, compatible CUDA toolkit/driver.
- BF16 or FP16 weights; **one GPU / `tensor_parallel_size=1`**. Gemma 1B has only
  one KV head, and this implementation does not replicate KV heads for TP.
- macOS and CPU can run the correctness tests. The inference engine requires CUDA.

The default context limit is 4,096 tokens. Gemma 1B's architecture allows 32,768,
but usable context also depends on the available KV cache. Prompts must leave room
for at least one generated token. Generation stops at a configured EOS, the output
budget, or the context limit, whichever comes first.

## Install

For CPU correctness tests on this Mac or another machine:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
TORCHDYNAMO_DISABLE=1 HF_HUB_OFFLINE=1 python -m unittest discover -s tests -v
```

For Linux CUDA inference, create a fresh environment, then install a compatible
PyTorch CUDA wheel before FlashAttention. This example uses CUDA 12.8; select the
wheel/toolkit appropriate for your host from [PyTorch's versioned instructions](https://pytorch.org/get-started/previous-versions/#v280).
FlashAttention may compile locally and needs the CUDA toolkit, C++ build tools,
and enough build memory; see its [installation requirements](https://github.com/Dao-AILab/flash-attention#installation-and-features).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[cuda]'
python -m pip install packaging ninja psutil
MAX_JOBS=4 python -m pip install flash-attn==2.8.3 --no-build-isolation
python -m pip check
```

The pinned reference stack is PyTorch 2.8.0, Transformers 4.57.3, Safetensors 0.6.2,
and Hugging Face Hub 0.36.0. FlashAttention is installed separately to make its
CUDA build prerequisites explicit.

## Download a checkpoint

Accept Google's model access terms on the relevant Hugging Face model page and
log in with that authorized account. Credentials and model weights are not part
of this repository.

```bash
hf auth login
hf download google/gemma-3-1b-pt --local-dir checkpoints/gemma-3-1b-pt
# For a chat model instead:
hf download google/gemma-3-1b-it --local-dir checkpoints/gemma-3-1b-it
```

Use a complete directory containing config, tokenizer, and `.safetensors` files.
A downloaded `config.json` alone is insufficient. To inspect metadata first:

```bash
python inspect_gemma3.py
```

## Generate

```bash
python example.py checkpoints/gemma-3-1b-pt --eager --prompt 'The capital of France is'
python example.py checkpoints/gemma-3-1b-it --chat --eager --prompt 'Explain KV caching simply.'
# Omit --eager to enable decode CUDA graphs.
```

```python
from minivllm import LLM, SamplingParams

with LLM('checkpoints/gemma-3-1b-pt', enforce_eager=True) as llm:
    outputs = llm.generate(
        ['The capital of France is', 'A GPU is useful for'],
        SamplingParams(temperature=0, max_tokens=64),
    )
    print(outputs[0]['text'])
```

`temperature=0` selects greedy decoding; positive temperature samples from the
full distribution. `max_tokens` must be positive. Outputs contain `text` and
`token_ids` in input order. Text omits special tokens; IDs retain a sampled EOS.
For IT chat, call `tokenizer.apply_chat_template(..., tokenize=True,
add_generation_prompt=True)` and pass the resulting token IDs; this avoids adding
BOS twice. Generation uses all EOS IDs in `generation_config.json`, including
Gemma IT's turn-ending token where configured.

`add_request(prompt, params)`, `step()`, and `is_finished()` expose incremental
scheduling. Only one engine should be alive per process. Use the context manager
or call `close()` to release its process group.

Useful engine options are `max_model_len`, `max_num_seqs`,
`max_num_batched_tokens`, `gpu_memory_utilization`, and `enforce_eager`.
`num_kvcache_blocks` optionally caps the automatic block count; this is useful
for small tests. The block size defaults to 256 and must be a multiple of 256.
`max_num_seqs` caps a scheduled batch; waiting/admitted requests can be more numerous.

## Verify before benchmarking

```bash
# No downloads; tiny random models, scheduler tests, and CPU reference attention.
TORCHDYNAMO_DISABLE=1 HF_HUB_OFFLINE=1 python -m unittest discover -s tests -v
# On CUDA: the same suite also runs real attention/cache and full-engine tests.
# Against your downloaded 1B checkpoint:
python verify_checkpoint.py checkpoints/gemma-3-1b-pt --eager
python verify_checkpoint.py checkpoints/gemma-3-1b-pt
# Measure this engine on your own hardware:
python bench.py checkpoints/gemma-3-1b-pt --requests 32 --input-tokens 256 --output-tokens 128
```

CPU tests compare custom model outputs with Transformers, including mixed prompt
lengths, local/global attention, cached/chunked execution, BF16/FP16 norm and RoPE
semantics, fused/tied weights, and malformed checkpoint rejection. They substitute
an independent attention oracle for the CUDA kernel. Scheduler and API tests cover
prefix sharing, eviction, page boundaries, preemption, limits, EOS, input validation,
and cleanup. The GPU tests use a locally created tiny checkpoint and require no
model-account access; they exercise real kernels, greedy generation, prefix reuse,
and eager versus CUDA-graph execution.

**Validation on the development Mac:** CPU checks and package installation pass.
CUDA tests are skipped here because NVIDIA hardware is unavailable. Real Gemma 1B
checkpoint generation, CUDA kernels/graphs, and throughput have not been measured
on this machine. There is no claimed speedup over vLLM or Transformers.

The benchmark reports actual generated tokens, synchronized wall time, repeated
runs, and median output tokens/s in ignored `results/benchmark.json`. It includes
prefill, sampling, and decoding text, excludes initialization/warmup, and uses fresh
synthetic prefixes per repetition. This is an offline throughput workload, not a
latency or serving benchmark.

## Code map

| File | Responsibility |
| --- | --- |
| `minivllm/models/gemma3.py` | Gemma decoder, local/global heads, embedding scaling, tied head |
| `minivllm/layers/` | Packed projections, offset RMSNorm, GELU, RoPE, FlashAttention, KV writes, sampler |
| `minivllm/engine/sequence.py` | Tokens, cache progress, page table, request state |
| `minivllm/engine/block_manager.py` | Physical pages, reference counts, chained prefix hashes, reuse |
| `minivllm/engine/scheduler.py` | Prefill-first batching, prompt chunks, decode, preemption, completion |
| `minivllm/engine/model_runner.py` | Packed inputs, GPU memory, execution context, CUDA graphs |
| `minivllm/engine/llm_engine.py` | Public generation and incremental request API |
| `minivllm/utils/loader.py` | Strict safetensors loading into packed/tied parameters |
| `tests/` | CPU reference, scheduler/API, and hardware-gated CUDA tests |

Gemma 3 requires offset RMSNorm (`1 + weight`), four decoder norms, gated tanh GELU,
`√hidden_size` embedding scaling, explicit `head_dim=256`, local/global RoPE bases
of 10,000/1,000,000, and five local layers followed by one global layer. A 512-token
local window maps to FlashAttention's inclusive `(511, 0)` bounds.

The cache retains full history for **all** layers, including local-attention layers.
Local attention still enforces the correct window, but hybrid sliding-window
storage reclamation is not implemented. BF16 Gemma 1B KV costs 26 KiB/token,
or 6.5 MiB per 256-token block across 26 layers. Multimodal models, quantization,
scaled RoPE variants, speculative decoding, production API serving, and multi-GPU
Gemma inference are outside this project's scope.

## Acknowledgments

This educational project builds on [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm).
See [UPSTREAM.md](UPSTREAM.md) for source attribution and implementation origins.

## License

MIT; the upstream copyright notice is retained. Model weights are governed by
Google's separate terms. Personal study notes are intentionally excluded from Git.
