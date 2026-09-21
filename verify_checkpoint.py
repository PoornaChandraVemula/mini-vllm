"""Compare real checkpoint greedy continuations with Hugging Face on CUDA."""
import argparse
import gc

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from minivllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model')
    parser.add_argument('--max-tokens', type=int, default=16)
    parser.add_argument('--eager', action='store_true')
    args = parser.parse_args()
    if args.max_tokens <= 0:
        parser.error('--max-tokens must be positive')
    if not torch.cuda.is_available():
        parser.error('This checkpoint validation requires an NVIDIA CUDA GPU')
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    prompts = [tokenizer.encode(text) for text in ['The capital of France is', 'A language model predicts']]
    reference = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype='auto', attn_implementation='eager',
    ).eval().cuda()
    generation = GenerationConfig(do_sample=False, max_new_tokens=args.max_tokens,
                                  eos_token_id=None, pad_token_id=tokenizer.pad_token_id)
    expected = []
    with torch.inference_mode():
        for prompt in prompts:
            ids = torch.tensor([prompt], device='cuda')
            output = reference.generate(input_ids=ids, attention_mask=torch.ones_like(ids), generation_config=generation)
            expected.append(output[0, len(prompt):].tolist())
    del reference
    gc.collect()
    torch.cuda.empty_cache()
    with LLM(args.model, enforce_eager=args.eager) as llm:
        actual = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=args.max_tokens, ignore_eos=True), use_tqdm=False)
    for index, (want, got) in enumerate(zip(expected, actual)):
        if want != got['token_ids']:
            raise AssertionError(f'Prompt {index}: HF {want} != mini-vllm {got["token_ids"]}. Inspect logits/precision before benchmarking.')
    print(f'PASS: {len(prompts)} greedy continuations match HF for {args.max_tokens} tokens each')


if __name__ == '__main__':
    main()
