"""Exercise actual bucketing, splitting, loader epochs and StageLR on CPU."""
import ast
from collections import defaultdict
import math
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.training_schedule import build_training_plan, make_stage_scheduler, stage_schedule_spec


def components():
    names = {'shuffle_with_seed', 'ConcatenatedBatchedDataset', 'Dataset', 'split_batch',
             'PipelineDataLoader', 'SkipFirstNSampler'}
    tree = ast.parse((ROOT / 'utils/dataset.py').read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    ns = dict(torch=torch, np=np, math=math, random=random, defaultdict=defaultdict,
              DEBUG=False, is_main_process=lambda: False)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<actual_dataset>', 'exec'), ns)
    return ns


class Leaf:
    def __init__(self, count, size, offset=0):
        self.count, self.size_bucket, self.offset = count, size, offset

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        return {'latents': torch.tensor([self.offset + index]), 'mask': None}


def dataset(ns, counts, rank=0, dp=8, gas=4, micro=16, subsample=None):
    ds = ns['Dataset'].__new__(ns['Dataset'])
    leaves, offset = [], 0
    for i, count in enumerate(counts):
        leaves.append(Leaf(count, (512 + i * 32, 512, 1), offset))
        offset += count
    ds.directory_datasets = [SimpleNamespace(get_size_bucket_datasets=lambda: leaves)]
    ds.dataset_config = {} if subsample is None else {'subsample_ratio': subsample}
    ds.post_init(rank, dp, {None: micro}, gas, {None: micro})
    return ds


def loader(ns, ds, gas=4):
    def prepare(batch, **kwargs):
        x = batch['latents']
        return (x, torch.empty(0)), (x, torch.empty(0))
    model = SimpleNamespace(prepare_inputs=prepare)
    engine = SimpleNamespace(is_pipe_parallel=False, is_first_stage=lambda: True, is_last_stage=lambda: True)
    return ns['PipelineDataLoader'](ds, engine, gas, model, num_dataloader_workers=0)


@pytest.mark.parametrize('mask_kind', ['none', 'empty', 'present'])
@pytest.mark.parametrize('gas', [1, 2, 4, 8])
def test_split_retains_every_sample(mask_kind, gas):
    ns = components()
    x = torch.arange(64).reshape(64, 1)
    mask = {'none': None, 'empty': torch.empty(0), 'present': torch.ones(64, 1)}[mask_kind]
    batches = ns['split_batch'](((x, torch.empty(0)), (x, mask, x + 10, x + 20)), gas)
    assert len(batches) == gas
    assert torch.equal(torch.cat([b[0][0] for b in batches]), x)
    assert torch.equal(torch.cat([b[1][2] for b in batches]), x + 10)
    assert all(len(b[1]) == 4 for b in batches)


def test_actual_mageflow_prepare_inputs_without_mask():
    from einops import rearrange
    import torch.nn.functional as F
    tree = ast.parse((ROOT / 'models/mage_flow.py').read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'prepare_inputs')
    ns = dict(torch=torch, rearrange=rearrange, F=F)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), '<MageFlow.prepare_inputs>', 'exec'), ns)
    model = SimpleNamespace(cache_text_embeddings=True, _uses_nl_variants=lambda: False,
                            _pad_cached_embeds=lambda embeds, device: (torch.zeros(64, 1, 4), torch.ones(64, 1)),
                            model_config={'timestep_sample_method': 'uniform'})
    batch = ns['prepare_inputs'](model, {'latents': torch.randn(64, 128, 2, 2),
                                         'mask': None, 'prompt_embeds': []})
    assert batch[1][1].numel() == 0
    microbatches = components()['split_batch'](batch, 4)
    assert len(microbatches) == 4
    assert torch.equal(torch.cat([b[1][0] for b in microbatches]), batch[1][0])


def test_invalid_batch_shapes_raise_instead_of_truncating():
    split = components()['split_batch']
    x = torch.ones(64, 1)
    with pytest.raises(ValueError, match='does not match'):
        split(((x,), (torch.ones(16, 1), None)), 4)
    with pytest.raises(ValueError, match='equal microbatches'):
        split(((torch.ones(63, 1),), (torch.ones(63, 1), None)), 4)


def test_reproduce_old_quarter_epoch_bug_and_fix():
    x = torch.arange(64).reshape(64, 1)
    old_labels = list(zip(torch.split(x, 16), torch.split(torch.empty(0), 16)))
    assert len(old_labels) == 1  # advertised four, actually one
    ns = components()
    # A concrete seven-bucket distribution with the user's exact image count.
    ds = dataset(ns, [5900] * 6 + [6198])
    assert sum(len(b.datasets[0]) for b in ds.buckets) == 41598
    assert len(ds) == 85
    dl = loader(ns, ds)
    plan = build_training_plan(ds, 10, 4)
    assert plan['steps_per_epoch'] == 85
    assert plan['total_steps'] == 850
    assert plan['microbatches_per_epoch'] == 340
    assert plan['scheduled_samples_per_epoch'] == 43520
    for step in range(1, 851):
        for _ in range(4):
            next(dl)
        assert dl.epoch == 1 + step // 85
    assert dl.epoch == 11


def test_all_eight_ranks_cover_every_image_with_empty_masks():
    ns = components()
    seen = set()
    for rank in range(8):
        ds = dataset(ns, [513, 63, 17], rank=rank)
        dl = loader(ns, ds)
        for _ in range(len(dl)):
            seen.update(next(dl)[0][0].flatten().tolist())
        assert dl.epoch == 2
    assert seen == set(range(593))


@pytest.mark.parametrize('consumed', [0, 1, 2, 3])
def test_checkpoint_before_inside_and_at_epoch_boundary(consumed):
    ns = components()
    ds = dataset(ns, [130], dp=1, gas=4, micro=16)
    first = loader(ns, ds)
    for _ in range(consumed * 4):
        next(first)
    state = first.state_dict()
    restored = loader(ns, ds)
    restored.load_state_dict(state)
    for _ in range(24):
        assert torch.equal(next(first)[0][0], next(restored)[0][0])
        assert first.epoch == restored.epoch
        assert first.state_dict() == restored.state_dict()


def test_legacy_boundary_checkpoint_does_not_replay_last_batch():
    ns = components()
    ds = dataset(ns, [1024])
    dl = loader(ns, ds)
    dl.load_state_dict({'epoch': 2, 'num_batches_pulled': 0})
    fresh = loader(ns, ds)
    assert torch.equal(next(dl)[0][0], next(fresh)[0][0])


def test_loader_detects_future_truncation():
    ns = components()
    dl = loader(ns, dataset(ns, [512]))
    dl.data = iter([((torch.ones(16, 1),), (torch.ones(16, 1),))])
    with pytest.raises(RuntimeError, match='truncated epoch'):
        next(dl)


def test_budget_subsample_small_buckets_max_steps_and_resume():
    ns = components()
    ds = dataset(ns, [1024, 17], subsample=0.5)
    plan = build_training_plan(ds, 10, 4, max_steps=7)
    assert plan['steps_per_epoch'] == 2
    assert plan['total_steps'] == 7
    ds = dataset(ns, [5900] * 6 + [6198])
    plan = build_training_plan(ds, 10, 4, completed_steps=90, epoch=2, batches_consumed=5)
    assert plan['total_steps'] == 850
    assert plan['remaining_steps'] == 760
    # Old broken run: 21 actual updates, now resuming at the start of epoch two.
    plan = build_training_plan(ds, 10, 4, completed_steps=21, epoch=2)
    assert plan['remaining_steps'] == 765
    assert plan['total_steps'] == 786


STAGES = [dict(type='linear', end_lr=7e-6, percent=0.1),
          dict(type='constant', lr=7e-6, percent=0.5),
          dict(type='rex', max_val=7e-6, min_val=0, percent=0.4)]


def test_stage_budget_warmup_rounding_and_manual_override():
    stages, budget, warmup, auto = stage_schedule_spec(dict(stages=STAGES, warmup_steps=50), 850)
    assert (budget, warmup, auto) == (800, 50, True)
    assert sum(round(s['percent'] * budget) for s in stages) == budget
    stages, budget, _, _ = stage_schedule_spec(dict(stages=STAGES), 7)
    assert [round(s['percent'] * budget) for s in stages] == [1, 4, 2]
    with pytest.warns(UserWarning, match='training ends at 850'):
        stage_schedule_spec(dict(stages=STAGES, total_iters=1487), 850)
    with pytest.raises(ValueError, match='sum to 1'):
        stage_schedule_spec(dict(stages=[dict(type='constant', lr=1, percent=0.2)]), 850)


def test_real_stagelr_boundaries_and_resume():
    StageLR = pytest.importorskip('stage_lr').StageLR
    cfg = dict(stages=STAGES, warmup_steps=50)
    p = torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.SGD([p], lr=1e-6)
    scheduler, _ = make_stage_scheduler(StageLR, opt, cfg, 850)
    assert scheduler.total_iters == 800
    assert [(s['start'], s['end']) for s in scheduler.stage_info] == [(0, 80), (80, 480), (480, 800)]
    sequence = []
    for _ in range(850):
        sequence.append(opt.param_groups[0]['lr'])
        opt.step()
        scheduler.step()
    assert sequence[129] == pytest.approx(7e-6)
    assert sequence[529] == pytest.approx(7e-6)
    assert sequence[-1] < sequence[530]
    for completed in (1, 49, 50, 129, 530, 849):
        resumed_opt = torch.optim.SGD([p], lr=1e-6)
        resumed, _ = make_stage_scheduler(StageLR, resumed_opt, cfg, 850,
                                          completed_steps=completed, base_lrs=[1e-6])
        assert resumed.last_epoch == completed
        assert resumed_opt.param_groups[0]['lr'] == pytest.approx(sequence[completed])
