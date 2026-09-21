# Provenance

mini-vllm adapts [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)
by Xingkai Yu, pinned at commit
[`bb823b3e06983d71485a8e1f23715ebd87d98ef8`](https://github.com/GeeeekExplorer/nano-vllm/tree/bb823b3e06983d71485a8e1f23715ebd87d98ef8).
The original MIT license and copyright notice are preserved in `LICENSE`.

The sequence/block manager, prefill-first scheduling, model runner, paged
FlashAttention integration, parallel projection layers, and context plumbing
are derived from that implementation. This repository uses the `minivllm`
package name and replaces Qwen3 with text-only Gemma 3.

Gemma-specific work includes its four-norm decoder layout, offset RMSNorm,
gated GELU, embedding scaling, local/global rotary embeddings, local attention
windows, tied checkpoint weights, multiple stop tokens, and explicit TP=1 scope.
Supporting fixes add strict loading, input/lifecycle checks, greedy sampling,
context/cache bounds, safe CPU staging tensors, bounded decode token batches,
and graph batch-size coverage. CPU reference and scheduler tests and optional
CUDA integration tests accompany the adaptation.

The implementation was checked against
[Transformers 4.57.3's Gemma3 model](https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/models/gemma3/modeling_gemma3.py)
and [Google's configuration](https://github.com/google/gemma_pytorch/blob/main/gemma/config.py).
No model weights or tokenizer assets are distributed here. Their terms are
separate from this project's MIT source-code license.
