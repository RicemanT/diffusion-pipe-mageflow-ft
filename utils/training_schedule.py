"""Training budgets derived from the exact, already-bucketed dataset iteration order."""
from collections import Counter
import math
import warnings


def positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{name} must be a positive integer, got {value!r}')
    return value


def build_training_plan(dataset, epochs, accumulation, *, max_steps=None,
                        completed_steps=0, epoch=1, batches_consumed=0):
    positive_int(epochs, 'epochs')
    positive_int(accumulation, 'gradient_accumulation_steps')
    steps_per_epoch = positive_int(len(dataset), 'processed dataset batches per epoch')
    if not 0 <= batches_consumed <= steps_per_epoch:
        raise ValueError('Checkpoint batch position is outside the processed epoch.')
    remaining = max(0, (epochs - epoch + 1) * steps_per_epoch - batches_consumed)
    if max_steps is not None:
        positive_int(max_steps, 'max_steps')
        remaining = min(remaining, max(0, max_steps - completed_steps))
    counts = Counter(i for i, _ in dataset.iteration_order)
    buckets = []
    for i, bucket in enumerate(dataset.buckets):
        source = bucket.num_samples_before_padding
        buckets.append(dict(
            size=list(bucket.datasets[0].size_bucket),
            samples_before_padding=source,
            padded_samples=len(bucket.iteration_order),
            added_samples=len(bucket.iteration_order) - source,
            requested_global_batch_size=bucket.requested_global_batch_size,
            global_batch_size=bucket.global_batch_size,
            batches_available=len(bucket), batches_scheduled=counts[i],
        ))
    return dict(
        epochs=epochs, steps_per_epoch=steps_per_epoch,
        microbatches_per_epoch=steps_per_epoch * accumulation,
        gradient_accumulation_steps=accumulation,
        data_parallel_world_size=dataset.data_parallel_world_size,
        samples_before_padding=sum(b['samples_before_padding'] for b in buckets),
        padded_samples=sum(b['padded_samples'] for b in buckets),
        scheduled_samples_per_epoch=sum(b['batches_scheduled'] * b['global_batch_size'] for b in buckets),
        epoch_budget_steps=epochs * steps_per_epoch,
        completed_steps=completed_steps, remaining_steps=remaining,
        total_steps=completed_steps + remaining, buckets=buckets,
    )


def print_training_plan(plan):
    print(f"Training plan: {plan['data_parallel_world_size']} data-parallel ranks, "
          f"accumulation={plan['gradient_accumulation_steps']}")
    print(f"  Processed sample slots before padding/repetition to fit batches: {plan['samples_before_padding']}")
    for b in plan['buckets']:
        print(f"  Bucket {b['size']}: samples={b['samples_before_padding']}, "
              f"padding/repeated={b['added_samples']}, global_batch={b['global_batch_size']} "
              f"(requested {b['requested_global_batch_size']}), steps={b['batches_scheduled']}")
    print(f"  Steps per epoch: {plan['steps_per_epoch']} optimizer steps = "
          f"{plan['microbatches_per_epoch']} microbatches per rank")
    print(f"  Epochs: {plan['epochs']}; full epoch budget: {plan['epoch_budget_steps']} steps; "
          f"planned final step: {plan['total_steps']}; remaining: {plan['remaining_steps']}")
    print(f"  Actual scheduled sample slots per epoch: {plan['scheduled_samples_per_epoch']} "
          '(includes bucket padding, configured repeats/resolutions and subsampling)')


def stage_schedule_spec(config, total_steps, default_warmup=0):
    warmup = config.get('warmup_steps', default_warmup)
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError('StageLR warmup_steps must be a nonnegative integer.')
    explicit = config.get('total_iters', config.get('total_steps'))
    automatic = explicit is None or explicit == 'auto'
    # Upstream StageLR adds warmup BEFORE its total_iters stage budget.
    budget = total_steps - warmup if automatic else positive_int(explicit, 'StageLR total_iters')
    positive_int(budget, 'StageLR steps after warmup')
    if not automatic and budget + warmup != total_steps:
        warnings.warn(f'Explicit StageLR schedule spans {budget + warmup} steps, but training ends at '
                      f'{total_steps}. Remove total_iters/total_steps or set it to "auto" to match training.')
    stages = config['stages']
    if not stages or len(stages) > budget:
        raise ValueError('StageLR needs at least one training step for each stage after warmup.')
    percents = [s['percent'] for s in stages]
    if any(not isinstance(p, (float, int)) or not math.isfinite(p) or p <= 0 for p in percents):
        raise ValueError('StageLR stage percentages must be finite positive numbers.')
    if not math.isclose(sum(percents), 1.0, rel_tol=0, abs_tol=1e-6):
        raise ValueError('StageLR stage percentages must sum to 1.')
    remaining = budget
    fitted = []
    for i, stage in enumerate(stages):
        count = remaining if i == len(stages) - 1 else min(
            max(1, round(stage['percent'] * budget)), remaining - (len(stages) - i - 1))
        fitted.append(dict(stage, percent=count / budget))
        remaining -= count
    return fitted, budget, warmup, automatic


def make_stage_scheduler(klass, optimizer, config, total_steps, default_warmup=0,
                         *, completed_steps=0, base_lrs=None):
    stages, budget, warmup, automatic = stage_schedule_spec(config, total_steps, default_warmup)
    if base_lrs is not None:
        for group, lr in zip(optimizer.param_groups, base_lrs):
            group['lr'] = lr
            group['initial_lr'] = lr
    scheduler = klass(optimizer, stages=stages, total_iters=budget, warmup_steps=warmup)
    if completed_steps:
        # Resume directly at the next update's LR without replaying optimizer steps
        # or importing stale stage boundaries from a differently-sized run.
        scheduler.last_epoch = completed_steps
        scheduler._step_count = completed_steps + 1
        lrs = scheduler.get_lr()
        for group, lr in zip(optimizer.param_groups, lrs):
            group['lr'] = lr
        scheduler._last_lr = lrs
    return scheduler, automatic


def print_stage_schedule(scheduler, automatic):
    print(f'StageLR: {"automatic" if automatic else "explicit"} schedule; '
          f'warmup={scheduler.warmup_steps}, stage steps={scheduler.total_iters}, '
          f'total={scheduler.warmup_steps + scheduler.total_iters}')
    if scheduler.warmup_steps:
        print(f'  warmup: optimizer updates 1..{scheduler.warmup_steps}')
    for stage in scheduler.stage_info:
        print(f"  {stage['type']}: optimizer updates "
              f"{scheduler.warmup_steps + stage['start'] + 1}..{scheduler.warmup_steps + stage['end']} "
              f"({stage['n_steps']} steps)")
