"""Measure actual generated tokens on a reproducible synthetic workload."""
import argparse
import json
from pathlib import Path
import random
import statistics
import time

import torch
from minivllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model')
    parser.add_argument('--requests', type=int, default=32)
    parser.add_argument('--input-tokens', type=int, default=256)
    parser.add_argument('--output-tokens', type=int, default=128)
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--eager', action='store_true')
    parser.add_argument('--json', type=Path, default=Path('results/benchmark.json'))
    args = parser.parse_args()
    if min(args.requests, args.input_tokens, args.output_tokens, args.repetitions) <= 0:
        parser.error('workload sizes and repetitions must be positive')
    rng = random.Random(args.seed)
    runs = []
    with LLM(args.model, enforce_eager=args.eager,
             max_model_len=args.input_tokens + args.output_tokens) as llm:
        if llm.config.max_model_len < args.input_tokens + args.output_tokens:
            raise ValueError('Requested workload exceeds available model/cache context capacity')
        params = SamplingParams(temperature=0.6, max_tokens=args.output_tokens, ignore_eos=True)
        llm.generate([[2]], SamplingParams(max_tokens=1), use_tqdm=False)
        for repetition in range(args.repetitions):
            # Fresh prompts avoid measuring a repeated prefix-cache hit workload.
            prompts = [[rng.randrange(3, llm.config.hf_config.vocab_size)
                        for _ in range(args.input_tokens)] for _ in range(args.requests)]
            torch.cuda.synchronize()
            start = time.perf_counter()
            outputs = llm.generate(prompts, params, use_tqdm=False)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            generated = sum(len(output['token_ids']) for output in outputs)
            if generated != args.requests * args.output_tokens:
                raise RuntimeError('Benchmark ended early: generated token count differs from workload')
            runs.append({'repetition': repetition + 1, 'seconds': elapsed,
                         'output_tokens': generated, 'output_tokens_per_second': generated / elapsed})
        report = {'model': str(Path(args.model).resolve()), 'gpu': torch.cuda.get_device_name(),
                  'torch': torch.__version__, 'cuda': torch.version.cuda,
                  'mode': 'eager' if args.eager else 'cuda_graph', 'seed': args.seed,
                  'requests': args.requests, 'input_tokens_per_request': args.input_tokens,
                  'output_tokens_per_request': args.output_tokens, 'runs': runs,
                  'median_output_tokens_per_second': statistics.median(r['output_tokens_per_second'] for r in runs),
                  'scope': 'End-to-end generation including prefill, sampling and text decoding; excludes model load, compilation and warmup. Synthetic tokens, fresh prefixes each run.'}
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
