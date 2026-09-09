"""Standalone CUDA attention benchmark; no weights, dataset, or DeepSpeed needed.

Example:
python tools/benchmark_mageflow_attention.py --transformer-config /models/Mage-Flow/transformer/config.json --backends sdpa flash_attn_3
"""

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.mageflow_attention import packed_attention, packed_attention_metadata, validate_attention_backend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--transformer-config', type=Path, required=True)
    parser.add_argument('--backends', nargs='+', default=['sdpa'], choices=['sdpa', 'flash_attn_2', 'flash_attn_3'])
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--height', type=int, default=1216)
    parser.add_argument('--width', type=int, default=832)
    parser.add_argument('--text-tokens', type=int, default=1280)
    parser.add_argument('--min-text-tokens', type=int, default=128)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--deterministic', action='store_true')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error('A CUDA GPU is required; CPU timings do not predict H100 performance.')
    if any(value <= 0 for value in (args.batch_size, args.height, args.width, args.text_tokens, args.iterations)):
        parser.error('batch size, dimensions, text tokens, and iterations must be positive')
    if args.height % 16 or args.width % 16:
        parser.error('height and width must be divisible by 16')
    if not 0 <= args.min_text_tokens <= args.text_tokens or args.warmup < 0:
        parser.error('invalid text length range or warmup')
    config = json.loads(args.transformer_config.read_text())
    heads = config['num_heads']
    dim = config['hidden_size'] // heads
    image_tokens = (args.height // 16) * (args.width // 16)
    length = args.text_tokens + image_tokens
    for backend in args.backends:
        validate_attention_backend(backend)

    torch.manual_seed(42)
    lengths = torch.linspace(args.min_text_tokens, args.text_tokens, args.batch_size,
                             device='cuda').round().int()
    text_mask = torch.arange(args.text_tokens, device='cuda')[None, :] < lengths[:, None]
    mask = torch.cat([text_mask, torch.ones(args.batch_size, image_tokens, device='cuda', dtype=torch.bool)], 1)
    mask = mask[:, None, None, :]
    q, k, v = [torch.randn(args.batch_size, heads, length, dim, device='cuda',
                          dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
    grad = torch.randn_like(q) * mask.transpose(-1, -2)

    print(json.dumps(dict(torch=torch.__version__, cuda=torch.version.cuda,
                          gpu=torch.cuda.get_device_name(), batch=args.batch_size,
                          heads=heads, head_dim=dim, image_tokens=image_tokens,
                          text_lengths=lengths.tolist(), dtype='bfloat16',
                          scope='one attention forward+backward including pack/unpack; excludes projections, MLP, Qwen, optimizer, communication')))

    metadata_times = []
    for _ in range(5):
        torch.cuda.synchronize()
        start = time.perf_counter()
        metadata = packed_attention_metadata(mask)
        torch.cuda.synchronize()
        metadata_times.append((time.perf_counter() - start) * 1000)
    print(json.dumps(dict(metadata_ms_median=statistics.median(metadata_times),
                          note='built once per microbatch; excluded from per-block timings below')))

    for backend in args.backends:
        def step():
            q.grad = k.grad = v.grad = None
            if backend == 'sdpa':
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            else:
                out = packed_attention(q, k, v, *metadata, backend, args.deterministic)
            out.backward(grad)

        for _ in range(args.warmup):
            step()
        q.grad = k.grad = v.grad = None
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(args.iterations):
            start = time.perf_counter()
            step()
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
        print(json.dumps(dict(backend=backend, forward_backward_ms_median=statistics.median(times),
                              min_ms=min(times), max_ms=max(times),
                              peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                              peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)))


if __name__ == '__main__':
    main()
