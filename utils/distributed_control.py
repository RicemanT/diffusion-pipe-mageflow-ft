"""Small integer collectives without object pickling or CUDA object gathers."""

import torch
import torch.distributed as dist


def _scalar(value):
    device = torch.device('cuda', torch.cuda.current_device()) if dist.get_backend() == 'nccl' else 'cpu'
    return torch.tensor(value, dtype=torch.int64, device=device)


def broadcast_int(value, src=0):
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return int(value)
    result = _scalar(value)
    dist.broadcast(result, src=src)
    return int(result.item())


def max_int(value):
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return int(value)
    result = _scalar(value)
    dist.all_reduce(result, op=dist.ReduceOp.MAX)
    return int(result.item())
