# AdamW8bitKahan correctness and precision

The inherited optimizer passed its `shift` compensation buffer as the parameter
to every bitsandbytes update kernel, while also passing `weight_decay`. This
decayed the residual, not the model weights. Weight decay was therefore incorrect,
not literally a no-op: it could erase part of the accumulated correction.
The fix adds `-lr * weight_decay * p` to the compensated update and passes zero
decay to all three kernels (32-bit state, blockwise 8-bit, legacy 8-bit).
Small decay updates consequently accumulate even below a BF16 weight's resolution.

`stabilize=True` is now rejected. Its previous calculation interpreted quantized
`state2` bytes as real second moments. The default was already false.
FP32 parameters now use the base bitsandbytes update without a compensation buffer.
The implementation follows the decay-through-compensation approach in Anzhc's
supplied example, retaining this repository's `shift` checkpoint key. It does not
adopt the example's separate asynchronous/foreach optimization.

## Stochastic rounding

The trainer does not provide universal stochastic rounding for optimizer updates
or gradient accumulation. Other optimizers already have their own implementations
(including Automagic, FFTDescent, OCGOptV2 and supported `adv_optm` optimizers).
AdamW8bitKahan previously had none.

MageFlow casts transformer parameters according to `model.dtype` and
`model.transformer_dtype` in `models/mage_flow.py`. Setting both to `bfloat16`
keeps trainable weights in BF16. Forward autocast does not create FP32 master
weights; `train.py`'s DeepSpeed configuration does not enable a mixed-precision
master-weight wrapper. Kahan compensation therefore matters for this setup.

To enable the new optimizer-local BF16 rounding:

```toml
[optimizer]
type = 'adamw8bitkahan'
lr = 5e-6                 # example; retain your chosen learning rate
betas = [0.9, 0.999]
weight_decay = 0.01
stabilize = false
stochastic_rounding = true
stochastic_rounding_seed = 0
force_kahan_buf_fp32 = false
```

The trainer already forwards these optimizer keys; no trainer-wide rounding
switch is required. With rounding enabled, gradient and compensated-update
working tensors are FP32, so bitsandbytes receives matching dtypes. The result
is rounded stochastically into the BF16 weight; its rounding error is retained
and stochastically stored in the BF16 residual. This changes optimizer writeback,
not gradient accumulation, communication precision, or quantized moment storage.
FP16 stochastic rounding is rejected; FP16 deterministic Kahan remains supported.

Rounding uses a local generator keyed by the configured seed, optimizer step,
parameter-group index and parameter index. Replicated parameters with identical
ordering receive identical rounding even when ranks' data/noise RNG streams
differ. Changing parameter ordering or sharding can change the rounding trajectory.
Keep the same rounding settings and seed when resuming.

Both new options default to false. `force_kahan_buf_fp32 = true` retains the
residual in FP32, adding two bytes per BF16 trainable element over the default
buffer (about 7.45 GiB for four billion elements). It can be combined with
stochastic rounding; FP32 residual storage then needs no stochastic cast.
Either option allocates FP32 working tensors per active parameter, with additional
rounding scratch for stochastic rounding. There is no persistent full-model FP32
master copy. Throughput and peak GPU memory have not been benchmarked.

## Compatibility and verification

Existing `shift` buffers are retained on resume and converted if the requested
buffer precision changes. Their dtype is protected during bitsandbytes checkpoint
loading. A missing buffer is initialized to zero. Historical incorrect decay cannot
be undone; continuation with nonzero decay will change the training trajectory.

Kahan parameters reject `max_unorm != 0` and `skip_zeros = true`, whose kernel
semantics do not match a compensated update. BF16 requires blockwise 8-bit state
and `percentile_clipping = 100`; use trainer gradient clipping where supported.
Legacy constructor options are passed through only on bitsandbytes versions that
accept them; otherwise the existing warning-and-ignore behavior is retained.

Run `python -m pytest test/test_adamw_8bit_kahan.py -q`. CPU tests check decay,
residual preservation, rounding statistics, replicated rounding and guards using
kernel stand-ins. Real bitsandbytes 0.50.2 CPU tests also check 32-bit state,
checkpoint continuation, and agreement with PyTorch AdamW using FP32 compensation.
CUDA tests exercise real small-tensor 32-bit states and large
blockwise 8-bit states, including checkpoint continuation. The development machine
has CPU-only PyTorch, so CUDA tests are skipped here.

The upstream [bitsandbytes optimizer implementation](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/bitsandbytes/optim/optimizer.py)
defines kernel dispatch and checkpoint casting behavior used by this subclass.
