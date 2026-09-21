"""Generate with a downloaded Gemma 3 1B PT or IT checkpoint."""
import argparse
from minivllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', help='Local checkpoint directory')
    parser.add_argument('--prompt', action='append', help='Repeat for a batch')
    parser.add_argument('--chat', action='store_true', help='Use the IT checkpoint chat template')
    parser.add_argument('--max-tokens', type=int, default=128)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--eager', action='store_true', help='Disable CUDA graph capture')
    args = parser.parse_args()
    prompts = args.prompt or ['The capital of France is', 'Explain how a KV cache helps language model inference.']
    with LLM(args.model, enforce_eager=args.eager) as llm:
        if args.chat:
            encoded = [llm.tokenizer.apply_chat_template(
                [{'role': 'user', 'content': prompt}], tokenize=True, add_generation_prompt=True,
            ) for prompt in prompts]
        else:
            encoded = prompts
        outputs = llm.generate(encoded, SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens))
        for prompt, output in zip(prompts, outputs):
            print(f'\nPrompt: {prompt}\nCompletion: {output["text"]}')


if __name__ == '__main__':
    main()
