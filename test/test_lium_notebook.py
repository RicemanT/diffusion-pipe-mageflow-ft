"""Offline notebook validation; never installs packages or launches GPUs."""
import ast
import io
import json
from pathlib import Path
import re
import tarfile
import urllib.request

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = json.loads((ROOT / 'DiffusionPipe_Lium_H100.ipynb').read_text(encoding='utf-8'))


def source(index):
    return ''.join(NOTEBOOK['cells'][index]['source'])


def archive(files):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode='w:gz') as tf:
        for name, text in files.items():
            data = text.encode()
            member = tarfile.TarInfo('repo/' + name)
            member.size = len(data)
            tf.addfile(member, io.BytesIO(data))
    return output.getvalue()


def fake_github(monkeypatch, revision='abc123', fail=False):
    files = {name: '' for name in ('train.py', 'utils/training_schedule.py',
                                  'utils/training_progress.py', 'optimizers/adamw_writeback.py')}
    files['.gitmodules'] = ('[submodule "comfy"]\npath = submodules/ComfyUI\n'
                           'url = https://github.com/test/comfy.git\n'
                           '[submodule "hyvideo"]\npath = submodules/HunyuanVideo\n'
                           'url = https://github.com/test/hyvideo.git\n')
    calls = []

    def urlopen(request, timeout):
        url = request.full_url
        calls.append(url)
        if url.endswith('/RicemanT/diffusion-pipe-mageflow-ft'):
            payload = {'default_branch': 'new/default'}
        elif '/commits/new%2Fdefault' in url:
            payload = {'sha': revision}
        elif '/git/trees/' in url:
            payload = {'tree': [{'type': 'commit', 'path': 'submodules/ComfyUI', 'sha': 'comfy123'},
                                {'type': 'commit', 'path': 'submodules/HunyuanVideo', 'sha': 'hy123'}]}
        elif '/test/comfy/' in url:
            return io.BytesIO(archive({'comfy/__init__.py': ''}))
        elif '/test/hyvideo/' in url:
            return io.BytesIO(archive({'wrong.txt' if fail else 'hyvideo/__init__.py': ''}))
        elif '/tar.gz/' + revision in url:
            return io.BytesIO(archive(files))
        else:
            raise AssertionError(url)
        return io.BytesIO(json.dumps(payload).encode())

    monkeypatch.setattr(urllib.request, 'urlopen', urlopen)
    return calls


def run_download(tmp_path, mode='update'):
    code = source(8).replace("MODE = 'update'", f'MODE = {mode!r}')
    exec(compile(code, '<download-cell>', 'exec'), {'LOCAL_ROOT': str(tmp_path)})


def test_all_cells_compile_and_outputs_are_cleared():
    for i, cell in enumerate(NOTEBOOK['cells']):
        if cell['cell_type'] == 'code':
            compile(source(i), f'cell-{i}', 'exec')
            assert cell['outputs'] == [] and cell['execution_count'] is None
    assert '40bf63a5' not in json.dumps(NOTEBOOK)


def test_download_follows_default_and_reuses_same_revision(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls = fake_github(monkeypatch)
    run_download(tmp_path)
    install = tmp_path / 'diffusion-pipe-mageflow-ft'
    assert (install / '.trainer-revision').read_text().strip() == 'abc123'
    assert any('/commits/new%2Fdefault' in url for url in calls)
    calls.clear()
    run_download(tmp_path)
    assert len(calls) == 2  # metadata + head; no redundant archives
    run_download(tmp_path, 'skip')
    assert len(calls) == 2


@pytest.mark.parametrize('fail', [True, False])
def test_update_preserves_old_install_and_failed_download_never_replaces_it(tmp_path, monkeypatch, fail):
    monkeypatch.chdir(tmp_path)
    install = tmp_path / 'diffusion-pipe-mageflow-ft'
    install.mkdir()
    (install / 'user-file').write_text('preserve')
    fake_github(monkeypatch, fail=fail)
    if fail:
        with pytest.raises(RuntimeError, match='HunyuanVideo'):
            run_download(tmp_path)
        assert (install / 'user-file').read_text() == 'preserve'
    else:
        run_download(tmp_path)
        backup = next(tmp_path.glob('*.backup-*'))
        assert (backup / 'user-file').read_text() == 'preserve'
        assert (install / '.trainer-revision').exists()


def test_progress_parser_accepts_trainer_totals_and_rejects_ds_steps():
    tree = ast.parse(source(20))
    names = {'step_log', 'plan_log'}
    nodes = [n for n in tree.body if isinstance(n, ast.Assign)
             and isinstance(n.targets[0], ast.Name) and n.targets[0].id in names]
    ns = {'re': re}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<parsers>', 'exec'), ns)
    line = ('epoch: 3/10  step: 213/850  epoch_step: 43/85  lr: 7.000e-06 / 1.000e-05  '
            'loss: 0.1234  12.00s/step  42.7img/s  elapsed: 2556s  ETA: 7644s  grad_norm: 0.1234')
    fields = ns['step_log'].search(line).groupdict()
    assert fields['step'] == '213' and fields['total'] == '850'
    assert fields['lr'] == '7.000e-06 / 1.000e-05'
    assert ns['step_log'].search(line.replace('img/s', 'sample/s'))
    assert not ns['step_log'].search('[Rank 0] step=852, skipped=0, lr=[0.001]')
    assert ns['plan_log'].search('planned final step: 786; remaining: 765').groups() == ('786', '765')


def test_launch_config_preserves_source_and_optimizer_settings(tmp_path):
    import toml
    original = {'epochs': 10, 'StageLR': {'total_iters': 1487, 'total_steps': 99,
                                        'stages': [{'type': 'linear', 'percent': 1.0}]},
                'optimizer': {'type': 'adamw8bitkahan', 'stochastic_rounding': True}}
    path = tmp_path / 'train.toml'
    path.write_text(toml.dumps(original))
    code = source(20)
    start = code.index('launch_config = toml.loads')
    end = code.index('\nif CACHE_ONLY and', start)
    ns = dict(toml=toml, Path=Path, _cfg_text=path.read_text(), CONFIG_PATH=str(path),
              AUTO_STAGE_BUDGET=True, RESUME_FROM_CHECKPOINT=True)
    exec(code[start:end], ns)
    assert toml.load(path) == original
    launched = toml.load(tmp_path / 'train.notebook.toml')
    assert launched['StageLR']['total_iters'] == 'auto'
    assert 'total_steps' not in launched['StageLR']
    assert launched['optimizer'] == original['optimizer']
    assert launched['resume_from_checkpoint'] is True
    assert launched['logging_steps'] == 1 and launched['progress_bar'] is False
