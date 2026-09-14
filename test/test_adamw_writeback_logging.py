"""Compile status reflects execution, including lazy failures and later fallback."""
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optimizers import adamw_writeback as module


def call(writeback):
    parameter = torch.ones(4, dtype=torch.bfloat16)
    writeback(parameter, torch.zeros(4), None, wide_update=True, stochastic=False, seed=0)


def test_success_only_after_first_execution_and_only_once(monkeypatch, capsys):
    def compile_fn(fn, **kwargs):
        assert capsys.readouterr().out == ''

        def execute(*args):
            assert 'active' not in capsys.readouterr().out
            return fn(*args)
        return execute

    monkeypatch.setattr(torch, 'compile', compile_fn)
    ticks = iter([10.0, 12.5])
    monkeypatch.setattr(module.time, 'perf_counter', lambda: next(ticks))
    writeback = module.Writeback(True)
    call(writeback)
    assert 'first call returned in 2.50s' in capsys.readouterr().out
    call(writeback)
    assert capsys.readouterr().out == ''


@pytest.mark.parametrize('lazy', [False, True])
def test_failure_never_reports_success(monkeypatch, capsys, lazy):
    def fail(*args, **kwargs):
        raise RuntimeError('compiler failure')
    monkeypatch.setattr(torch, 'compile', (lambda *a, **kw: fail) if lazy else fail)
    writeback = module.Writeback(True)
    with pytest.warns(UserWarning, match='using eager writeback'):
        call(writeback)
    call(writeback)
    assert capsys.readouterr().out == ''
    assert not writeback.success_reported


def test_later_failure_still_warns_after_initial_success(monkeypatch, capsys):
    monkeypatch.setattr(torch, 'compile', lambda fn, **kw: fn)
    writeback = module.Writeback(True)
    call(writeback)
    assert 'compiled writeback active' in capsys.readouterr().out

    def fail(*args):
        raise RuntimeError('new shape failed')
    writeback.compiled = fail
    with pytest.warns(UserWarning, match='using eager writeback'):
        call(writeback)
    assert capsys.readouterr().out == ''
    assert writeback.failed


def test_other_ranks_and_eager_mode_are_quiet(monkeypatch, capsys):
    call(module.Writeback(False))
    monkeypatch.setattr(torch, 'compile', lambda fn, **kw: fn)
    monkeypatch.setattr(torch.distributed, 'is_available', lambda: True)
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: True)
    monkeypatch.setattr(torch.distributed, 'get_rank', lambda: 1)
    call(module.Writeback(True))
    assert capsys.readouterr().out == ''
