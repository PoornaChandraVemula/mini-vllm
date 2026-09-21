"""Inspect Gemma configuration/tensor metadata without downloading full weights."""
import argparse
import json
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='google/gemma-3-1b-pt')
    parser.add_argument('--output', type=Path, default=Path('checkpoints/gemma-3-1b-pt'))
    args = parser.parse_args()
    api = HfApi()
    revision = api.model_info(args.model).sha
    if not revision:
        raise RuntimeError('Could not resolve the checkpoint revision')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'revision.txt').write_text(revision + '\n')
    config = hf_hub_download(args.model, 'config.json', revision=revision, local_dir=args.output)
    print('Checkpoint revision:', revision)
    print(json.dumps(json.loads(Path(config).read_text()), indent=2))
    metadata = api.get_safetensors_metadata(args.model, revision=revision)
    for filename, file_metadata in metadata.files_metadata.items():
        print('\nFILE:', filename)
        for name, tensor in file_metadata.tensors.items():
            print(name, tensor.shape, tensor.dtype)


if __name__ == '__main__':
    main()
