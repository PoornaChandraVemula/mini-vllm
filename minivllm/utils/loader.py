"""Load HF safetensors into fused projections, rejecting incomplete checkpoints."""
from pathlib import Path
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    if param.shape != loaded_weight.shape:
        raise ValueError(f'Checkpoint shape {tuple(loaded_weight.shape)} != parameter shape {tuple(param.shape)}')
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    files = sorted(Path(path).glob('*.safetensors'))
    if not files:
        raise FileNotFoundError(f'No .safetensors weights found in {path}; config.json alone is not a checkpoint')
    mapping = getattr(model, 'packed_modules_mapping', {})
    parameters = dict(model.named_parameters())
    aliases = {name: next(n for n, p in parameters.items() if p is param)
               for name, param in model.named_parameters(remove_duplicate=False)}
    required = set()
    for name in parameters:
        shards = [shard for target, shard in mapping.values() if f'.{target}.' in name]
        required.update((name, shard) for shard in shards) if shards else required.add((name, None))
    loaded = set()
    seen_names = set()
    for file in files:
        with safe_open(file, framework='pt', device='cpu') as f:
            for name in f.keys():
                if name in seen_names:
                    raise ValueError(f'Duplicate checkpoint tensor: {name}')
                seen_names.add(name)
                target, shard = name, None
                for source, (packed, part) in mapping.items():
                    if f'.{source}.' in name:
                        target = name.replace(f'.{source}.', f'.{packed}.')
                        shard = part
                        break
                if target not in aliases:
                    raise ValueError(f'Unexpected checkpoint tensor: {name}')
                canonical = aliases[target]
                param = parameters[canonical]
                weight = f.get_tensor(name)
                expected_shape = list(param.shape)
                if shard is not None:
                    module = model.get_submodule(target.rsplit('.', 1)[0])
                    if isinstance(shard, str):
                        heads = module.num_heads if shard == 'q' else module.num_kv_heads
                        expected_shape[0] = heads * module.head_size
                    else:
                        expected_shape[0] = module.output_sizes[shard]
                if tuple(weight.shape) != tuple(expected_shape):
                    raise ValueError(f'Invalid shape for checkpoint tensor {name}: {tuple(weight.shape)}; expected {tuple(expected_shape)}')
                loader = getattr(param, 'weight_loader', default_weight_loader)
                if (canonical, shard) in loaded:
                    # Some serializers include both tied embedding and output weights.
                    if not torch.equal(param.detach().cpu(), weight.to(param.dtype)):
                        raise ValueError(f'Conflicting tied checkpoint tensor: {name}')
                    continue
                try:
                    if shard is None:
                        loader(param, weight)
                    else:
                        loader(param, weight, shard)
                except (RuntimeError, AssertionError, IndexError) as exc:
                    raise ValueError(f'Invalid shape for checkpoint tensor {name}: {tuple(weight.shape)}') from exc
                loaded.add((canonical, shard))
    missing = required - loaded
    if missing:
        names = ', '.join(f'{name}[{shard}]' if shard is not None else name for name, shard in sorted(missing, key=str))
        raise ValueError(f'Checkpoint is missing required weights: {names}')
