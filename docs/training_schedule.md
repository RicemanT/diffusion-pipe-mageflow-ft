# Exact epoch counts and StageLR budgets

For eight data-parallel GPUs, microbatch 16 per GPU and accumulation 4:

```
effective global batch = 8 * 16 * 4 = 512
41598 / 512 = 81.24609375
```

A single bucket with its remainder kept needs **82 optimizer steps per epoch**,
or **820 steps for ten epochs**. Dropping the remainder would give 81/810, but this
trainer pads bucket tails. Pipeline-parallel stages are not additional data-parallel
replicas: always use the actual data-parallel world size in that multiplication.

The actual trainer count is the sum of fitted batches across resolution/aspect-ratio/
frame buckets, after repeats, expanded cached samples and subsampling. Tails are
padded per bucket. Small buckets can use reduced batches, and image/video or
per-resolution batch-size overrides also affect the count. Raw image count alone
cannot determine the exact bucketed result.

For example, seven buckets containing `[5900, 5900, 5900, 5900, 5900, 5900, 6198]`
sum to exactly 41,598 images and yield **85 batches**, or **850 steps for ten epochs**.
This is an illustrative distribution, not a claim about the contents of a remote
dataset. The trainer now reports the actual bucket sizes and counts itself.

## Why the old run could show 21.25 steps per epoch

MageFlow returns `torch.empty(0)` for an absent mask. The old `split_batch` used
`torch.split` on every tensor and combined the results with `zip`. Splitting the
empty mask returns **one chunk**, while splitting 64 local images for accumulation
4 returns four chunks. `zip` silently truncates to one.

The loader consequently emitted only 85 microbatches instead of 340, discarding
the final three image chunks from every dataset batch. Each optimizer update then
pulled four *different* dataset batches, crossing epoch boundaries:

```
85 emitted microbatches / 4 per update = 21.25 updates per logical epoch
```

Actual updates are integer-valued; epoch boundaries are observed at alternating
21/22-update intervals, and ten epochs would finish after about 213 updates.
Meanwhile the old scheduler length calculation still advertised 850. This was a
data-loss bug, not just a misleading progress counter.

Empty/absent masks and optional features now appear in **every** microbatch.
Nonempty tensors must match the input batch dimension. Invalid splits fail loudly.
The loader verifies that each epoch emits exactly its advertised microbatch count;
the trainer also verifies engine/loader accumulation agreement.

## Automatic planning and StageLR

Use a top-level scheduler selection and omit the manual total:

```toml
epochs = 10
micro_batch_size_per_gpu = 16
gradient_accumulation_steps = 4
pipeline_stages = 1
lr_scheduler = 'StageLR'

[StageLR]
# total_iters = 'auto'  # optional; omission means the same thing
stages = [
    { type = 'linear', end_lr = 7e-6, percent = 0.1 },
    { type = 'constant', lr = 7e-6, percent = 0.5 },
    { type = 'rex', max_val = 7e-6, min_val = 0, percent = 0.4 },
]
```

The trainer calculates the exact count from the processed dataset before the first
training update. It prints every bucket's source/padded sample counts, effective
batch and scheduled steps, the optimizer steps and microbatches per epoch, total
and remaining steps, and the actual StageLR update ranges. It writes
`training_plan.json` in the run directory and includes the plan in new checkpoints.
`max_steps` caps the run and the automatically derived schedule.

With 85 batches and no separate warmup, the example above allocates:

| Stage | Training update numbers | Count |
| --- | --- | --- |
| Linear | 1–85 | 85 |
| Constant | 86–510 | 425 |
| REX | 511–850 | 340 |

Stage percentages must sum to one. Integer rounding is fitted to the exact budget
with at least one update per stage and any remainder assigned to the final stage.

The external [StageLR implementation](https://github.com/nruaif/stage-lr/blob/master/stage_lr/scheduler.py)
adds a separate `warmup_steps` phase *before* its `total_iters` stages. Automatic
planning therefore passes `total_training_steps - warmup_steps` as the stage
budget. For an 850-update run with 50 warmup updates, StageLR gets 800 stage updates,
not 850 + 50. A linear first stage does not imply a separate warmup unless you
configure `warmup_steps` (including a top-level warmup setting).

An explicit numeric `total_iters`/`total_steps` retains its existing meaning as a
post-warmup stage budget. The trainer warns if it differs from the actual run.
Remove old hand-calculated values such as `1487` to enable automatic planning.
This change preserves the library's LR curve formulas, including its REX endpoint
convention; it corrects their timing rather than redefining the curves.

Progress logs now include, for example:

```
epoch: 2/10  step: 100/850  epoch_step: 15/85  lr: ...  loss: ...
```

TensorBoard and the configured tracker receive `train/total_steps`,
`train/steps_per_epoch`, `train/epoch_step`, and `train/progress` alongside loss/LR.

## Live training progress

Interactive terminals now show a moving completion bar on rank zero, with a
second status row. For example:

```text
Epoch 3/10  25%|████████                        | 213/850 [elapsed 00:42:36 | ETA 02:07:24]
ep 43/85 | loss 0.1234 | lr 7e-06 | 12.00s/step 0.08step/s | 42.7img/s
```

The total comes from the same exact training plan used by StageLR, including
`max_steps` and the remaining work on resume. The bar advances once per optimizer
update, not per microbatch. LR is captured before that update so it reports the
rate actually used, even when the scheduler advances immediately afterward.
Different group LRs appear as a range; individual group rates remain in tracking.

Set these options at the **top level** of the training TOML, before any table:

```toml
progress_bar = true  # default: animated in a terminal, ordinary lines in log files
# progress_bar = "always"  # force animation for consoles that do not report a TTY
# progress_bar = false     # ordinary periodic console lines
# steps_per_print = 100    # optional: reduce DeepSpeed's separate periodic messages
```

Loss, epoch position, seconds/step, steps/second, and images/second update on the
bar independently of `logging_steps`. Speed is a rolling average of the latest
50 optimizer updates, including data loading and the training call. Throughput
counts global sample slots across data-parallel ranks, including bucket padding
and repeats; reduced buckets contribute their actual batch size. Video or mixed
datasets use `sample/s` instead of `img/s`. The example counter used by
`x_axis_examples` likewise accumulates actual batch sizes.

Elapsed time starts with the training loop and includes saving, validation, and
sampling. ETA extrapolates the current session's average wall time per update;
it improves after observing those pauses, but cannot predict unseen future
overhead. Setup, caching, and the pre-training evaluation/sampling baseline are
excluded. Checkpoints persist elapsed time up to the start of their save; older
checkpoints without that field start a fresh elapsed timer. Resumed speed and ETA
use newly measured updates instead of counting historical steps as new work.

Saving, validation, and sampling get explicit phase labels. Ordinary Python
prints and console logging appear above the bar, and completion or interruption
closes it cleanly. The same timing metrics are sent to TensorBoard and the
configured tracker at `logging_steps`: `train/seconds_per_step`,
`train/steps_per_second`, `train/samples_per_second`, `train/elapsed_seconds`, and
`train/eta_seconds`. With animation disabled, periodic text lines retain the
counts, loss, LR, throughput, elapsed time, and ETA.

## Preview the training plan

To print the plan without executing training updates, append this flag to the
usual command:

```bash
deepspeed --num_gpus=8 train.py --config YOUR_CONFIG.toml --print_training_plan
```

This still loads the model and prepares/loads dataset caches. It does not perform
training, validation or sampling, and never requires a complete trial training run.

## Resume and verification

New loader checkpoints store consumed batches explicitly instead of inferring
them from lookahead-prefetched batches. Legacy counters are supported, including
the epoch-boundary zero counter that previously could become a negative skip.
StageLR is rebuilt against the current planned final update and positioned at the
completed update count; stale saved stage lengths do not override the new plan.
Configuration provides the schedule definition and starting learning rates.

Resuming an old shortened run cannot recover images already skipped. It keeps the
saved epoch/loader position and counts the remaining work correctly from there.
For example, 21 completed updates and a resume at the start of epoch 2 with 85
batches leaves 765 updates, giving a planned final global step of 786. Resetting
the loader or changing the dataset likewise changes the remaining budget. For
comparable full-data experiments, start a fresh run with the corrected splitter.

CPU regressions use the actual dataset bucket/split/loader classes, simulate all
eight data-parallel ranks, verify every sample is retained, and traverse the full
850-update/ten-epoch case. They exercise mid-epoch and boundary resumes, reduced
buckets, subsampling, explicit overrides, integer stage rounding and real StageLR
warmup/REX boundaries. Full multi-H100 DeepSpeed execution remains untested locally.
