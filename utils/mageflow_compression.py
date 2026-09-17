"""Native support for Bluvoll's mageflow-lowrank-modulation-v1 format.

The shared input projection is owned by the initial pipeline layer. Each block
owns its own two heads; only the projected activation travels between layers.
"""

import json
from pathlib import Path

import safetensors
import torch
from torch import nn
import torch.nn.functional as F


ARCHITECTURE = 'mageflow-lowrank-modulation-v1'


def compressed_parameter(name):
    return name.startswith('modulation_down.') or (
        name.startswith('transformer_blocks.')
        and ('.img_mod.' in name or '.txt_mod.' in name)
    )


class ModulationProjection(nn.Module):
    """An explicitly typed projection, excluded from generic Linear adapters.

    Parameters and gradients retain their selected precision on device moves.
    In particular, never cast through BF16 and then back to FP32: that loses
    calibrated coefficients. Autocast is disabled for the projection itself.
    """

    def __init__(self, in_features, out_features, dtype=torch.float32, device=None):
        super().__init__()
        self.parameter_dtype = dtype
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype, device=device))
        self.bias = nn.Parameter(torch.empty(out_features, dtype=dtype, device=device))

    def _apply(self, fn, recurse=True):
        def preserve(tensor):
            if not tensor.is_floating_point():
                return fn(tensor)
            destination = fn(tensor.new_empty(0)).device
            return tensor.to(device=destination, dtype=self.parameter_dtype)
        return super()._apply(preserve, recurse=recurse)

    def forward(self, x):
        with torch.autocast(x.device.type, enabled=False):
            return F.linear(x.to(self.weight.dtype), self.weight, self.bias)


def read_transformer_config(path):
    """Read compressed metadata without loading multi-GB tensor payloads."""
    with safetensors.safe_open(path, framework='pt', device='cpu') as f:
        metadata = f.metadata() or {}
        has_down = 'modulation_down.weight' in f.keys()
        architecture = metadata.get('architecture')
        if architecture == ARCHITECTURE:
            try:
                config = json.loads(metadata['model_config'])
            except (KeyError, ValueError) as exc:
                raise ValueError('Compressed MageFlow requires valid model_config metadata') from exc
            rank = config.get('modulation_rank')
            hidden = config.get('hidden_size')
            if type(rank) is not int or type(hidden) is not int or not 0 < rank <= hidden:
                raise ValueError('Invalid compressed MageFlow modulation_rank/hidden_size')
            if not has_down or f.get_slice('modulation_down.weight').get_shape() != [rank, hidden]:
                raise ValueError('Compressed MageFlow shared projection does not match metadata')
            return config
        if has_down or architecture is not None:
            raise ValueError(f'Unsupported MageFlow checkpoint architecture: {architecture!r}')
    with open(Path(path).parent / 'config.json', encoding='utf-8') as f:
        config = json.load(f)
    if config.get('modulation_rank', 0):
        raise ValueError('Compressed MageFlow requires architecture and model_config safetensors metadata')
    return config


def install_compressed_modulation(transformer, rank, dtype=torch.float32):
    if dtype not in (torch.float32, torch.bfloat16):
        raise ValueError('compressed_adaln_dtype must be float32 or bfloat16')
    hidden = transformer.inner_dim
    device = transformer.img_in.weight.device
    transformer.modulation_down = ModulationProjection(hidden, rank, dtype, device)
    for block in transformer.transformer_blocks:
        block.compressed_modulation = True
        for stream in ('img_mod', 'txt_mod'):
            # SiLU is applied once before the shared down projection.
            setattr(block, stream, nn.Sequential(
                nn.Identity(), ModulationProjection(rank, 6 * hidden, dtype, device)))


def resolve_modulation_dtype(model_config):
    value = model_config.get('compressed_adaln_dtype', 'float32')
    if value in ('float32', torch.float32):
        return torch.float32
    if value in ('bfloat16', torch.bfloat16):
        return torch.bfloat16
    raise ValueError('model.compressed_adaln_dtype must be float32 or bfloat16')
