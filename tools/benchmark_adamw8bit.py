"""Small-update accuracy, optimizer state size and warmed step timing.

Run from the repository root. CUDA exercises real 8-bit states; CPU uses BNB's
32-bit-state fallback. This is an optimizer microbenchmark, not a quality eval.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optimizers.adamw_8bit import AdamW8bitKahan
import bitsandbytes as bnb


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def run(args, mode, compiled):
    device = torch.device(args.device)
    p = torch.nn.Parameter(torch.ones(args.numel, device=device, dtype=torch.bfloat16))
    p.grad = torch.full_like(p, 0.1)
    options = dict(lr=args.lr, weight_decay=args.weight_decay,
                   min_8bit_size=4096 if device.type == 'cuda' else 2**60)
    if mode == 'nearest':
        opt = bnb.optim.AdamW8bit([p], **options)
    else:
        opt = AdamW8bitKahan([p], **options, kahan_sum=mode != 'sr_only',
                            stochastic_rounding=mode in ('kahan_sr', 'sr_only'),
                            force_kahan_buf_fp32=mode == 'kahan_fp32',
                            compile_writeback=compiled)
    synchronize(device)
    cold_start = time.perf_counter()
    opt.step()
    synchronize(device)
    cold_seconds = time.perf_counter() - cold_start
    for _ in range(args.warmup):
        opt.step()
    synchronize(device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for _ in range(args.steps):
        opt.step()
    synchronize(device)
    seconds = time.perf_counter() - start
    reference = 1.0
    grad = p.grad[0].float().item()
    for _ in range(1 + args.warmup + args.steps):
        reference = reference * (1 - args.lr * args.weight_decay) - args.lr * grad / (abs(grad) + 1e-8)
    error = p.detach().float() - reference
    state = opt.state[p]
    residual = state.get('shift')
    result = dict(
        mode=mode, compiled_requested=compiled,
        compiled_active=compiled and not opt._writeback.failed,
        cold_step_seconds=cold_seconds, warmed_step_ms=seconds * 1000 / args.steps,
        reference=reference, mean_weight=p.float().mean().item(),
        weight_rmse=error.square().mean().sqrt().item(),
        unchanged_fraction=(p == 1).float().mean().item(),
        state_bytes=sum(t.numel() * t.element_size() for t in state.values() if isinstance(t, torch.Tensor)),
        residual_bytes=0 if residual is None else residual.numel() * residual.element_size(),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
    )
    if residual is not None:
        represented = p.float() + residual.float()
        result['compensated_rmse'] = (represented - reference).square().mean().sqrt().item()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--numel', type=int, default=16384)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--compile', action='store_true', help='Also attempt real Inductor writeback')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.steps < 1 or args.numel < 1 or args.warmup < 0:
        parser.error('steps and numel must be positive; warmup must be nonnegative')
    if args.device == 'cpu':
        torch.set_num_threads(1)
    result = dict(torch=torch.__version__, bitsandbytes=bnb.__version__,
                  device=args.device, numel=args.numel, steps=args.steps,
                  warmup=args.warmup, lr=args.lr, weight_decay=args.weight_decay,
                  note='CPU uses 32-bit states; timings include BNB synchronization and RNG.', results=[])
    for mode in ('nearest', 'kahan', 'kahan_sr', 'sr_only', 'kahan_fp32'):
        for compiled in ([False, True] if args.compile and mode != 'nearest' else [False]):
            row = run(args, mode, compiled)
            result['results'].append(row)
            print(json.dumps(row), flush=True)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
