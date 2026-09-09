"""Checkpoint requests must lead every rank through identical collectives."""

import ast
from datetime import timedelta
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.distributed_control import broadcast_int, max_int


def saver_class(is_main, broadcast):
    tree = ast.parse((ROOT / 'utils/saver.py').read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Saver')
    ns = dict(is_main_process=lambda: is_main, broadcast_int=broadcast,
              need_to_checkpoint=lambda config: False, sys=sys,
              logger=SimpleNamespace(warning=lambda *args: None))
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<Saver>', 'exec'), ns)
    saver = ns['Saver'].__new__(ns['Saver'])
    saver.config = {}
    return saver


@pytest.mark.parametrize('signal', ['save', 'save_quit'])
def test_rank_zero_broadcasts_and_consumes_signal_after_checkpoint(tmp_path, signal):
    path = tmp_path / signal
    path.touch()
    received = []
    saver = saver_class(True, lambda value: received.append(value) or value)
    saver.save_root = tmp_path
    saved = []

    def save(step, examples):
        assert path.is_file(), 'do not lose the request before a successful checkpoint'
        saved.append((step, examples))

    saver.save_checkpoint = save
    if signal == 'save_quit':
        with pytest.raises(SystemExit):
            saver.process_step(12, 6144)
    else:
        assert saver.process_step(12, 6144) == (True, False)
    assert received == [2 if signal == 'save_quit' else 1]
    assert saved == [(12, 6144)]
    assert not path.exists()


def test_failed_checkpoint_keeps_request(tmp_path):
    (tmp_path / 'save').touch()
    saver = saver_class(True, lambda value: value)
    saver.save_root = tmp_path

    def fail(*args):
        raise OSError('disk full')

    saver.save_checkpoint = fail
    with pytest.raises(OSError, match='disk full'):
        saver.process_step(1, 512)
    assert (tmp_path / 'save').exists()


@pytest.mark.parametrize('signal', [0, 1, 2])
def test_other_ranks_follow_broadcast_without_filesystem_reads(signal):
    class ForbiddenPath:
        def __truediv__(self, name):
            return self

        def is_file(self):
            raise AssertionError('nonzero rank checked signal files')

    saver = saver_class(False, lambda value: signal)
    saver.save_root = ForbiddenPath()
    saved = []
    saver.save_checkpoint = lambda *args: saved.append(args)
    if signal == 2:
        with pytest.raises(SystemExit):
            saver.process_step(1, 512)
    else:
        assert saver.process_step(1, 512) == (signal == 1, False)
    assert len(saved) == (signal != 0)


def test_quit_request_takes_priority(tmp_path):
    (tmp_path / 'save').touch()
    (tmp_path / 'save_quit').touch()
    received = []
    saver = saver_class(True, lambda value: received.append(value) or value)
    saver.save_root = tmp_path
    saver.save_checkpoint = lambda *args: None
    with pytest.raises(SystemExit):
        saver.process_step(1, 512)
    assert received == [2]


def collective_worker(rank, rendezvous):
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    try:
        assert broadcast_int(2 if rank == 0 else 0) == 2
        assert max_int(3 + rank) == 4
        assert broadcast_int(0) == 0
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason='requires Gloo')
def test_integer_collectives_across_two_processes(tmp_path):
    mp.spawn(collective_worker, args=((tmp_path / 'rendezvous').as_uri(),), nprocs=2, join=True)
