"""Tensor-only AdamW writeback, separate from bitsandbytes and RNG management."""
import time
import warnings

import torch


def _round_bf16(source, noise):
    bits = source.contiguous().view(torch.int32)
    values = ((bits + noise) & -65536).view(torch.float32)
    ordinary = source.to(torch.bfloat16)
    values = torch.where(torch.isfinite(source) & torch.isfinite(ordinary), values, source)
    return values.to(torch.bfloat16)


@torch.no_grad()
def _copy_bf16_stochastic_(target, source, generator=None):
    noise = torch.empty_like(source, dtype=torch.int32).random_(0, 65536, generator=generator)
    target.copy_(_round_bf16(source, noise))


def writeback_values(parameter, update, residual, wide_update, noise, residual_noise):
    """Return new values without mutating inputs, so compile failures can retry safely.

    The update already includes the old residual and real-weight decay. Explicit
    dtype boundaries preserve the legacy low-precision Kahan arithmetic.
    """
    if not wide_update:
        new_parameter = (parameter + update).to(parameter.dtype)
        difference = (parameter - new_parameter).to(parameter.dtype)
        return new_parameter, (update + difference).to(residual.dtype)

    previous = parameter.float()
    total = previous + update
    new_parameter = _round_bf16(total, noise) if noise is not None else total.to(parameter.dtype)
    if residual is None:
        return new_parameter, None
    new_residual = update + (previous - new_parameter.float())
    if residual_noise is not None:
        new_residual = _round_bf16(new_residual, residual_noise)
    else:
        new_residual = new_residual.to(residual.dtype)
    return new_parameter, new_residual


class Writeback:
    """Generate rank-consistent noise eagerly; optionally compile only pure math.

    BNB kernels, Python seeds and parameter copies stay outside the graph.
    Failure retries the pure function with the same noise, never the Adam step.
    """
    def __init__(self, compiled=False):
        self.requested = compiled
        self.compiled = None
        self.failed = False
        self.success_reported = False

    @torch.no_grad()
    def __call__(self, parameter, update, residual, *, wide_update, stochastic, seed):
        noise = residual_noise = None
        if stochastic:
            generator = torch.Generator(device=parameter.device).manual_seed(seed)
            noise = torch.empty_like(update, dtype=torch.int32).random_(0, 65536, generator=generator)
            if residual is not None and residual.dtype == torch.bfloat16:
                residual_noise = torch.empty_like(update, dtype=torch.int32).random_(0, 65536, generator=generator)
        args = (parameter, update, residual, wide_update, noise, residual_noise)
        first_call_started = None
        if self.requested and not self.failed:
            if not self.success_reported:
                first_call_started = time.perf_counter()
            try:
                if self.compiled is None:
                    # Inductor normally removes intermediate BF16 round trips.
                    # Kahan relies on observing those exact rounding errors.
                    self.compiled = torch.compile(
                        writeback_values, fullgraph=True, dynamic=True,
                        options={'emulate_precision_casts': True},
                    )
                new_parameter, new_residual = self.compiled(*args)
            except Exception as exc:
                self.failed = True
                warnings.warn(f'AdamW writeback compilation failed; using eager writeback: {exc}', stacklevel=2)
                new_parameter, new_residual = writeback_values(*args)
        else:
            new_parameter, new_residual = writeback_values(*args)
        parameter.copy_(new_parameter)
        if residual is not None:
            residual.copy_(new_residual)
        if first_call_started is not None and not self.failed:
            self.success_reported = True
            distributed = torch.distributed
            if not distributed.is_available() or not distributed.is_initialized() or distributed.get_rank() == 0:
                elapsed = time.perf_counter() - first_call_started
                print(
                    f'AdamW compiled writeback active: first call returned in {elapsed:.2f}s '
                    '(includes compilation; additional shapes may compile later).',
                    flush=True,
                )
