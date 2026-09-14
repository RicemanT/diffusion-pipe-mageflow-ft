"""Progress accounting and terminal cleanup without DeepSpeed or GPUs."""
import ast
from contextlib import nullcontext
import io
import logging
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.training_progress import TrainingProgress, batch_samples, duration


class Clock:
    value = 0.0

    def __call__(self):
        return self.value


def update(progress, step=1, samples=512, seconds=2):
    return progress.update(step, 1, step, 85, loss=0.1234, lrs=[7e-6],
                           samples=samples, seconds=seconds)


def test_resume_rates_use_only_new_steps_and_eta_includes_overhead():
    clock = Clock()
    with TrainingProgress(850, 10, initial=800, elapsed=1000, enabled=False, clock=clock) as p:
        clock.value = 2
        stats = update(p, 801)
        assert stats['train/seconds_per_step'] == 2
        assert stats['train/steps_per_second'] == 0.5
        assert stats['train/samples_per_second'] == 256
        assert stats['train/elapsed_seconds'] == 1002
        assert stats['train/eta_seconds'] == 98
        with p.phase('saving'):
            clock.value += 10
        clock.value += 2
        stats = update(p, 802, samples=128)
        assert stats['train/samples_per_second'] == 160
        assert stats['train/seconds_per_step'] == 2
        assert stats['train/eta_seconds'] == 336  # 14 / 2 * 48
        assert stats['train/elapsed_seconds'] == 1014


def test_throughput_window_excludes_old_slow_steps():
    with TrainingProgress(51, 1, enabled=False) as p:
        update(p, seconds=1000)
        for step in range(2, 52):
            stats = update(p, step)
        assert stats['train/seconds_per_step'] == 2
        assert stats['train/eta_seconds'] == 0


def test_actual_bucket_samples_include_reduced_batch_and_resume_position():
    dataset = SimpleNamespace(iteration_order=[(0, 0), (1, 0), (0, 1)],
                              buckets=[SimpleNamespace(global_batch_size=512),
                                       SimpleNamespace(global_batch_size=128)])
    assert [batch_samples(dataset, i) for i in range(3)] == [512, 128, 512]


@pytest.mark.parametrize('main,enabled', [(False, 'always'), (True, False), (True, True)])
def test_noninteractive_or_other_rank_does_not_emit_control_codes(main, enabled):
    output = io.StringIO()
    with TrainingProgress(2, 1, main_process=main, enabled=enabled, file=output) as p:
        update(p)
        with p.phase('validating'):
            pass
    assert output.getvalue() == ''


@pytest.mark.parametrize('fail', [False, True])
def test_live_bar_renders_and_restores_output_and_loggers(fail):
    output = io.StringIO()
    original_stdout = sys.stdout
    handlers = logging.getLogger().handlers[:]
    p = TrainingProgress(2, 10, enabled='always', file=output)
    with pytest.raises(RuntimeError) if fail else nullcontext():
        with p:
            update(p)
            with p.phase('validating'):
                print('Validation message above the bar')
            if fail:
                raise RuntimeError('test interruption')
            update(p, 2)
    text = output.getvalue()
    assert '\r' in text
    assert 'Epoch 1/10' in text
    assert 'elapsed' in text and 'ETA' in text
    assert ('Stopped' in text) == fail
    assert p.bar.n == (1 if fail else 2)
    assert sys.stdout is original_stdout
    assert logging.getLogger().handlers == handlers
    assert 'loss 0.1234' in p.details
    assert 'lr 7e-06' in p.details
    assert '256.0img/s' in p.details


def test_lr_range_and_video_units():
    with TrainingProgress(1, 1, enabled=False, image_only=False) as p:
        p.update(1, 1, 1, 1, loss=0.1, lrs=[1e-5, 7e-6], samples=8, seconds=2)
    assert '7e-06..1e-05' in p.details
    assert '4.0sample/s' in p.details


def test_rejects_duplicate_or_excess_steps():
    with TrainingProgress(1, 1, enabled=False) as p:
        update(p)
        with pytest.raises(ValueError):
            update(p)
        with pytest.raises(ValueError):
            update(p, 2)


def test_duration():
    assert duration(None) == '--:--:--'
    assert duration(360001) == '100:00:01'


@pytest.mark.parametrize('has_progress', [True, False])
def test_checkpoint_records_elapsed_and_closes_saving_phase(has_progress):
    tree = ast.parse((ROOT / 'utils/saver.py').read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Saver')
    ns = {'nullcontext': nullcontext}
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<Saver>', 'exec'), ns)
    saver = ns['Saver'].__new__(ns['Saver'])
    clock = Clock()
    p = TrainingProgress(10, 1, elapsed=100, clock=clock, enabled=False)
    saver.train_dataloader = SimpleNamespace(state_dict=lambda: {'epoch': 1})
    if has_progress:
        saver.train_dataloader.training_progress = p
    recorded = []
    saver.model_engine = SimpleNamespace(save_checkpoint=lambda *a, **kw: recorded.append(kw))
    saver.save_root = 'unused'
    clock.value = 20
    saver.save_checkpoint(3, 1536)
    state = recorded[0]['client_state']
    assert state['training_elapsed_seconds'] == (120 if has_progress else 0)
    assert state['step'] == 3 and state['examples'] == 1536


def test_progress_wraps_training_only_and_includes_final_save():
    tree = ast.parse((ROOT / 'train.py').read_text())
    evaluate = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'evaluate_single')
    assert not any(isinstance(n, ast.With) for n in ast.walk(evaluate))
    main = next(n for n in tree.body if isinstance(n, ast.If) and '__name__' in ast.unparse(n.test))
    progress = next(n for n in main.body if isinstance(n, ast.With) and ast.unparse(n.items[0].context_expr) == 'progress')
    assert isinstance(progress.body[0], ast.While)
    calls = [n.func.attr for n in ast.walk(progress) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    assert 'train_batch' in calls and 'save_checkpoint' in calls and 'save_model' in calls


def test_actual_training_loop_counts_buckets_and_logs_lr_before_scheduler_step():
    tree = ast.parse((ROOT / 'train.py').read_text())
    main = next(n for n in tree.body if isinstance(n, ast.If) and '__name__' in ast.unparse(n.test))
    loop = next(n for n in main.body if isinstance(n, ast.With) and ast.unparse(n.items[0].context_expr) == 'progress')
    clock = Clock()
    loader = SimpleNamespace(epoch=1, micro_batches_consumed=0, gradient_accumulation_steps=4,
                             sync_epoch=lambda: None)
    optimizer = SimpleNamespace(param_groups=[{'lr': 0.1}])

    def train_batch(iterator):
        clock.value += 2
        loader.micro_batches_consumed += 4
        if loader.micro_batches_consumed == 8:
            loader.micro_batches_consumed = 0
            loader.epoch += 1
        optimizer.param_groups[0]['lr'] *= 0.5
        return SimpleNamespace(item=lambda: 0.25)

    logs, saves = [], []
    saver = SimpleNamespace(
        process_epoch=lambda epoch, step, examples: (loader.epoch if loader.epoch <= 2 else None, False, False),
        process_step=lambda *a: (False, False),
        save_checkpoint=lambda step, examples: saves.append(('checkpoint', step, examples)),
        save_model=lambda name: saves.append(('model', name)),
    )
    dataset = SimpleNamespace(iteration_order=[(0, 0), (1, 0)],
                              buckets=[SimpleNamespace(global_batch_size=512),
                                       SimpleNamespace(global_batch_size=128)])
    ns = dict(
        progress=TrainingProgress(4, 2, enabled=False, clock=clock),
        time=SimpleNamespace(perf_counter=clock), batch_samples=batch_samples,
        train_dataloader=loader, train_data=dataset, optimizer=optimizer,
        _get_param_group_lrs=lambda opt: [opt.param_groups[0]['lr']], is_main_process=lambda: True,
        get_data_iterator_for_step=lambda *args: None,
        model_engine=SimpleNamespace(reset_activation_shape=lambda: None, train_batch=train_batch),
        step=1, epoch=1, examples=0, epoch_loss=0, num_steps=0, total_training_steps=4,
        steps_per_epoch=2, saver=saver,
        config=dict(epochs=2, x_axis_examples=True, logging_steps=1, log_lr_to_console=False,
                    eval_every_n_steps=None, eval_every_n_epochs=None),
        tb_writer=SimpleNamespace(add_scalar=lambda name, value, axis: logs.append((name, value, axis))),
        tracker=SimpleNamespace(enabled=False), sampling_config=None,
        validation_sampling=SimpleNamespace(should_sample=lambda *args: False),
    )
    exec(compile(ast.Module(body=[loop], type_ignores=[]), '<training-loop>', 'exec'), ns)
    lr_logs = [(value, axis) for name, value, axis in logs if name == 'train/lr']
    assert lr_logs == [(0.1, 512), (0.05, 640), (0.025, 1152), (0.0125, 1280)]
    assert ns['progress'].completed == 4
    assert ns['progress'].metrics()['train/samples_per_second'] == 160
    assert saves == [('checkpoint', 4, 1280), ('model', 'epoch2')]
