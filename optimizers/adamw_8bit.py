import inspect
import warnings

import torch
import bitsandbytes
import bitsandbytes.functional as F


# Keys that older bitsandbytes exposed via get_config()/__init__ but which
# newer versions removed entirely:
#   - percentile_clipping (and its state["gnorm_vec"] buffer)
#   - block_wise          (the non-blockwise 8-bit path was dropped)
# Passing them to the constructor now raises TypeError, and reading them out
# of get_config() now raises KeyError, so this optimizer tolerates both the
# old and new bitsandbytes rather than pinning to one.
_LEGACY_BNB_KWARGS = ('percentile_clipping', 'block_wise')


class AdamW8bitKahan(bitsandbytes.optim.AdamW8bit):
    """AdamW8bit with Kahan summation compensation.

    The 'shift' buffer is the Kahan compensation term: bitsandbytes writes the
    parameter update into `shift` instead of directly into `p`, then the tail
    of update_step folds it into `p` while carrying the lost low-order bits
    forward. That recovers most of the precision lost by keeping master weights
    in bf16, which is what makes bf16 training viable without an fp32 copy.

    Because Kahan compensation is intrinsic to this class, there is no
    `kahan_sum` toggle -- it is always on. A `kahan_sum` kwarg is accepted and
    ignored (with a warning if someone tries to disable it) so that configs
    written for optimi-style optimizers don't hard-crash here.

    stochastic_rounding enables BF16 parameter and residual writeback from
    per-parameter FP32 working tensors. force_kahan_buf_fp32 instead (or also)
    retains the residual in FP32. Both default to False. FP32 parameters use
    ordinary bitsandbytes AdamW. See docs/adamw8bitkahan.md for costs and resume.
    """

    def __init__(self, *args, stabilize=False, kahan_sum=None,
                 stochastic_rounding=False, force_kahan_buf_fp32=False,
                 stochastic_rounding_seed=0, **kwargs):
        if stabilize:
            raise ValueError('stabilize=True is unsupported: quantized state2 is not a real second moment.')
        if kahan_sum is False:
            warnings.warn(
                'AdamW8bitKahan: kahan_sum=false was requested but Kahan summation is '
                'intrinsic to this optimizer and cannot be disabled. Use type = '
                "'adamw8bit' if you want plain AdamW8bit without compensation.")
        # Retain legacy options only when the installed BNB accepts them.
        for key in _LEGACY_BNB_KWARGS:
            if key in kwargs and key not in inspect.signature(bitsandbytes.optim.AdamW8bit).parameters:
                warnings.warn(
                    f'AdamW8bitKahan: ignoring "{key}", which current bitsandbytes '
                    f'no longer supports.')
                kwargs.pop(key)
        super().__init__(*args, **kwargs)
        self.stabilize = False
        self.stochastic_rounding = stochastic_rounding
        self.force_kahan_buf_fp32 = force_kahan_buf_fp32
        self.stochastic_rounding_seed = int(stochastic_rounding_seed)
        self.non_castable_tensor_keys.add('shift')

    def _shift_dtype(self, p):
        return torch.float32 if self.force_kahan_buf_fp32 else p.dtype

    @torch.no_grad()
    def init_state(self, group, p, gindex, pindex):
        super().init_state(group, p, gindex, pindex)
        if p.dtype in (torch.bfloat16, torch.float16):
            self.state[p]['shift'] = self.get_state_buffer(p, dtype=self._shift_dtype(p))

    @torch.no_grad()
    def update_step(self, group, p, gindex, pindex):
        if p.dtype not in (torch.bfloat16, torch.float16):
            return super().update_step(group, p, gindex, pindex)
        # avoid update error from non-contiguous memory layout
        p.data = p.data.contiguous()
        p.grad = p.grad.contiguous()

        state = self.state[p]
        grad = p.grad

        config = self.get_config(gindex, pindex, group)

        if config.get('max_unorm', 0.0) != 0.0 or config.get('skip_zeros', False):
            raise ValueError('Kahan compensation requires max_unorm=0 and skip_zeros=False.')
        if p.dtype == torch.bfloat16 and state['state1'].dtype == torch.uint8 and not config.get('block_wise', True):
            raise ValueError('BF16 Kahan requires block_wise=True for 8-bit state.')
        if self.stochastic_rounding and p.dtype != torch.bfloat16:
            raise ValueError('stochastic_rounding is supported only for BF16 parameters.')
        if config.get('percentile_clipping', 100) < 100 and p.dtype == torch.bfloat16:
            raise ValueError('BF16 Kahan requires percentile_clipping=100; use trainer gradient clipping.')

        # Preserve old checkpoint residuals, including when changing buffer precision.
        if 'shift' not in state:
            state['shift'] = torch.zeros_like(p, dtype=self._shift_dtype(p))
        else:
            state['shift'] = state['shift'].to(device=p.device, dtype=self._shift_dtype(p))

        state["step"] += 1
        step = state["step"]

        # Percentile clipping only exists on older bitsandbytes, and needs the
        # gnorm_vec buffer that newer versions no longer allocate. Guard on
        # both so this is a no-op (gnorm_scale = 1.0, matching what modern
        # bitsandbytes hardcodes) instead of a KeyError.
        percentile_clipping = config.get("percentile_clipping", 100)
        if percentile_clipping < 100 and "gnorm_vec" in state:
            current_gnorm, clip_value, gnorm_scale = F.percentile_clipping(
                grad,
                state["gnorm_vec"],
                step,
                percentile_clipping,
            )
        else:
            gnorm_scale = 1.0

        wide_update = self.stochastic_rounding or self.force_kahan_buf_fp32
        shift = state['shift'].float() if wide_update else state['shift']
        # BNB dispatches on gradient dtype and expects the update tensor to match.
        if wide_update:
            grad = grad.float()
        lr = config['lr']
        # Decay the real weights through compensation; never decay the residual.
        if config['weight_decay'] != 0:
            shift.add_(p.detach(), alpha=-lr * config['weight_decay'])

        if state["state1"].dtype == torch.float:
            F.optimizer_update_32bit(
                self.optimizer_name,
                grad,
                shift,
                state["state1"],
                config["betas"][0],
                config["eps"],
                step,
                lr,
                state["state2"],
                config["betas"][1],
                config["betas"][2] if len(config["betas"]) >= 3 else 0.0,
                config.get("alpha", 0.0),
                0.0,
                gnorm_scale,
                state["unorm_vec"] if config["max_unorm"] > 0.0 else None,
                max_unorm=config["max_unorm"],
                skip_zeros=config["skip_zeros"],
            )

        elif state["state1"].dtype == torch.uint8 and not config.get("block_wise", True):
            # Legacy non-blockwise 8-bit path. Only reachable on older
            # bitsandbytes; modern versions removed both the config key and
            # the max1/max2/new_max1/new_max2 state buffers this needs.
            F.optimizer_update_8bit(
                self.optimizer_name,
                grad,
                shift,
                state["state1"],
                state["state2"],
                config["betas"][0],
                config["betas"][1],
                config["eps"],
                step,
                lr,
                state["qmap1"],
                state["qmap2"],
                state["max1"],
                state["max2"],
                state["new_max1"],
                state["new_max2"],
                0.0,
                gnorm_scale=gnorm_scale,
                unorm_vec=state["unorm_vec"] if config["max_unorm"] > 0.0 else None,
                max_unorm=config["max_unorm"],
            )

            # swap maxes
            state["max1"], state["new_max1"] = state["new_max1"], state["max1"]
            state["max2"], state["new_max2"] = state["new_max2"], state["max2"]

        elif state["state1"].dtype == torch.uint8:
            # Blockwise 8-bit: the only 8-bit path modern bitsandbytes keeps.
            F.optimizer_update_8bit_blockwise(
                self.optimizer_name,
                grad,
                shift,
                state["state1"],
                state["state2"],
                config["betas"][0],
                config["betas"][1],
                config["betas"][2] if len(config["betas"]) >= 3 else 0.0,
                config.get("alpha", 0.0),
                config["eps"],
                step,
                lr,
                state["qmap1"],
                state["qmap2"],
                state["absmax1"],
                state["absmax2"],
                0.0,
                gnorm_scale=gnorm_scale,
                skip_zeros=config["skip_zeros"],
            )

        else:
            raise RuntimeError(f"Unsupported optimizer state dtype: {state['state1'].dtype}")

        # Fold the update into p and retain the error from parameter writeback.
        if wide_update:
            previous = p.float()
            updated = previous + shift
            if self.stochastic_rounding:
                # Rank-local training RNG differs with data/noise. Use identical
                # rounding on replicated parameters, reproducible after resume.
                generator = torch.Generator(device=p.device)
                generator.manual_seed((self.stochastic_rounding_seed + step * 1000003
                                       + gindex * 1000033 + pindex * 1000037) % (2**63))
                _copy_bf16_stochastic_(p.data, updated, generator)
            else:
                p.data.copy_(updated)
            shift.add_(previous.sub_(p.float()))
            if self.stochastic_rounding and state['shift'].dtype == torch.bfloat16:
                _copy_bf16_stochastic_(state['shift'], shift, generator)
            else:
                state['shift'].copy_(shift)
        else:
            buffer = p.clone()
            p.add_(shift)
            shift.add_(buffer.sub_(p))


@torch.no_grad()
def _copy_bf16_stochastic_(target, source, generator=None):
    """Unbiased FP32 -> BF16 rounding; preserve NaN/Inf and finite overflow casts."""
    bits = source.contiguous().view(torch.int32)
    rounded = torch.empty_like(bits).random_(0, 65536, generator=generator)
    rounded.add_(bits).bitwise_and_(-65536)
    values = rounded.view(torch.float32)
    # Do not turn NaNs into infinities or wrap values at the exponent boundary.
    ordinary = source.to(torch.bfloat16)
    values = torch.where(torch.isfinite(source) & torch.isfinite(ordinary), values, source)
    target.copy_(values)
