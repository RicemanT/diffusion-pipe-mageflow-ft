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
Keep the same rounding settings and seed when resuming. New checkpoints record
the Kahan mode, stochastic rounding setting and seed, and reject mismatches before
loading. Legacy checkpoints without that metadata remain supported. Gradient
release assigns distinct seeds to its per-parameter optimizers; older gradient
release runs without seed metadata may change rounding trajectory after this update.

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

## Optional SR-only mode and compiled writeback

Keep `type = 'adamw8bitkahan'` for all of these modes. Defaults remain Kahan on,
stochastic rounding off, and compiled writeback off. Existing runs do not switch
to SR-only automatically.

| Mode | `kahan_sum` | `stochastic_rounding` | Persistent residual per BF16 element |
| --- | --- | --- | --- |
| Existing default Kahan | `true` | `false` | 2 bytes |
| Kahan plus SR | `true` | `true` | 2 bytes |
| Experimental SR-only | `false` | `true` | None |

`kahan_sum = false` now explicitly selects SR-only (previously it was ignored
with a warning). It requires `stochastic_rounding = true` and cannot be combined
with `force_kahan_buf_fp32 = true`. SR-only still computes Adam updates in FP32
with bitsandbytes moment storage and the same decoupled decay. It rounds the
resulting weight without storing a residual. It does not change Adam's epsilon,
clipping, betas or moment quantization rules.

Standalone SR is useful: a small update that always disappears under nearest
rounding instead has a proportional probability of moving to the adjacent BF16
value. Its average follows the intended update, with extra variance in individual
weights. Kahan explicitly accumulates rounding error and can be combined with SR.
Neither mechanism guarantees better diffusion quality; see the discussion of
both approaches in [Revisiting BFloat16 Training](https://arxiv.org/abs/2010.06192).
For the main training workflow, retain Kahan. SR-only is an experimental memory
option: removing a BF16 residual saves about 7.45 GiB for four billion trainable
elements per full replica, though temporary FP32 work still consumes memory.

Start SR-only with fresh optimizer state. Loading a Kahan checkpoint into SR-only
is rejected rather than discarding its residual. New SR-only checkpoints likewise
reject a change of numerical mode. Compiler settings can change on resume;
`force_kahan_buf_fp32` can still change while retaining/converting Kahan residuals.

To try compilation, add this to the existing `[optimizer]` section:

```toml
compile_writeback = true
```

`optimizers/adamw_8bit.py` owns state management and BNB dispatch;
`optimizers/adamw_writeback.py` owns tensor-only writeback and rounding. This
separation is inspired by SDNQ's organization; the mathematical implementation
retains this fork's AdamW and decay behavior.

Compilation covers writeback only, not BNB kernels, random-number generation,
gradient accumulation, or the trainer. Random integers are generated outside the
graph using the same local seed in eager and compiled modes. Inductor is asked to
preserve intermediate precision casts, which Kahan requires. Compiled execution
returns new values before copying them into parameters/state. A compile failure
warns once per writeback instance and switches that instance to eager, reusing the
same noise and never repeating the Adam update. First-use compilation adds startup
cost; dtype/rank/shape variants may need additional graphs. Older PyTorch versions
without the required compiler option fall back with a warning.

After the first successful compiled call and parameter writeback, rank zero prints
`AdamW compiled writeback active: first call returned in ...s` once per writeback
instance. This is host wall time including first-use compilation, not a synchronized
GPU benchmark. It adds no CUDA synchronization. Further shapes may still compile;
any later compilation failure still warns and switches that instance to eager.
Disabled compilation and failed first attempts do not print a success message.

## Reproducing the evaluation

```bash
python tools/benchmark_adamw8bit.py --device cuda --numel 1048576 --steps 1000 --compile --output adamw8bit_gpu_benchmark.json
```

This uses actual BNB updates to compare nearest rounding, Kahan, Kahan+SR,
SR-only, and FP32 Kahan buffers. It reports cold-step time, warmed full optimizer
step time, state/residual bytes, CUDA peak allocated memory, and error on constant
small updates against an analytical FP32 AdamW reference. Compilation fallback is
explicitly reported as `compiled_active = false`. Compilation is not a claimed
speedup until the warmed GPU timings improve. For larger models, also profile
end-to-end training: this single-tensor test excludes pipeline/block swapping,
communication, and the model's tensor-size distribution.

CPU runs use 32-bit BNB states because the CPU backend does not provide the same
CUDA 8-bit kernels. They establish numerical behavior, not GPU throughput or
diffusion quality. The checked-in CPU result records its versions and configuration.

### Local numerical results

[Recorded CPU results](adamw8bit_cpu_benchmark.json), PyTorch 2.14.0+cpu and
bitsandbytes 0.50.2, used 16,384 identical BF16 weights initially equal to 1,
constant gradient 0.1, LR 5e-6, and weight decay 0.01. After 1,011 total steps
(one cold, ten warmup, 1,000 timed), the reference weight was **0.99489458**.

| Mode | Mean weight | Per-weight RMSE against reference | Residual bytes |
| --- | --- | --- | --- |
| Nearest BF16 | 1.00000000 | 0.00510544 | 0 |
| BF16 Kahan | 1.00000000 | 0.00510544 | 32,768 |
| BF16 Kahan + SR | 0.99489021 | 0.00180356 | 32,768 |
| SR-only | 0.99492145 | 0.00444658 | 0 |
| FP32 Kahan buffer | 0.99609375 | 0.00119919 | 65,536 |

This deliberately small-update case also exposes a limitation of a deterministic
BF16 Kahan buffer: the residual itself can stop accumulating once its rounding
spacing grows too large for an incoming update. It is not a new regression from
the refactor. Kahan+SR avoided that stall and had less per-weight error than
SR-only here; SR-only tracked the mean but introduced more individual-weight
variance. The FP32 buffer retained the represented weight (weight plus residual)
to about 6e-8 RMSE, despite its visible BF16 weight's rounding error.

These findings support keeping Kahan and evaluating `stochastic_rounding = true`
for BF16 training, with SR-only reserved for memory pressure. They do not establish
convergence or image quality on MageFlow. Current config defaults remain unchanged.

Every real Inductor attempt on this Windows machine fell back because `cl` was
unavailable. The graph-capture tests pass with AOT eager, but compiled CPU/CUDA
kernel parity and speed are not established locally. CPU timing variations between
the fallback and eager rows are not compiler speedups. The CUDA regression tests
require actual compiled/eager equality and do not count fallback as success.
