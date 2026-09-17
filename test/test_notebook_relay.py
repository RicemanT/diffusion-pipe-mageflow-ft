"""Offline coverage for the portable notebook staging/transfer worker."""
import io
import json
from pathlib import Path
import sys
import tarfile
from types import SimpleNamespace

import pytest

from utils import notebook_relay as relay


def test_standalone_excludes_only_base_weights():
    for filename in ('transformer/model.safetensors', 'transformer/nested/model.bin',
                     'transformer/model.safetensors.index.json'):
        assert not relay.selected_base(filename, 'standalone')
        assert relay.selected_base(filename, 'full')
    for filename in ('transformer/config.json', 'vae/model.safetensors',
                     'text_encoder/model-00001-of-00002.safetensors', 'tokenizer/tokenizer.json'):
        assert relay.selected_base(filename, 'standalone')
    assert relay.checkpoint_directory('RicemanT/MageTrail', 'main',
        'V0.2/MageTrail-V0.2-Epoch100.safetensors').endswith('/48cb93dfc1f3')


@pytest.mark.parametrize('name', ['../escape', '/absolute', 'a/../../escape', 'a\\escape'])
def test_manifest_paths_cannot_escape(name):
    with pytest.raises(ValueError):
        relay.safe_relative(name)


def test_balanced_lanes_cover_every_file_once():
    entries = [{'path': f'f{i}', 'size': size} for i, size in enumerate([100, 80, 50, 40, 30, 20, 10])]
    lanes = relay.balanced_lanes(entries, 4)
    names = [name for lane in lanes for name in lane]
    assert sorted(names) == sorted(e['path'] for e in entries)
    assert len(lanes) == 4
    assert all(lanes)


def captions_archive(unsafe=False):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as archive:
        for i in range(1002):
            entry = tarfile.TarInfo(f'artist/{i}_nl.txt')
            entry.size = 1
            archive.addfile(entry, io.BytesIO(b'x'))
        if unsafe:
            entry = tarfile.TarInfo('escape')
            entry.type = tarfile.SYMTYPE
            entry.linkname = '/etc/passwd'
            archive.addfile(entry)
    return buf.getvalue()


def test_unsafe_sidecar_links_rejected():
    with tarfile.open(fileobj=io.BytesIO(captions_archive(True)), mode='r:gz') as archive:
        with pytest.raises(ValueError, match='Unexpected'):
            relay.sidecar_members(archive)


def test_staging_pins_revisions_preserves_paths_and_never_copies_configs(tmp_path, monkeypatch):
    checkpoint = 'V0.2/MageTrail-V0.2-Epoch100.safetensors'
    repositories = {
        'base': {name: b'{}' for name in (
            'model_index.json', 'transformer/config.json', 'vae/config.json',
            'text_encoder/config.json', 'transformer/model.safetensors',
            'vae/model.safetensors', 'text_encoder/model.safetensors')},
        'custom': {checkpoint: b'weights'},
        'cache': {'metadata/grouping_keys.json': b'{}',
                  'metadata/grouped_metadata_0.arrow': b'cached-metadata',
                  'latents/0.arrow': b'latents', 'sidecars.tar.gz': captions_archive()},
    }
    calls = []

    class Api:
        def __init__(self, **kwargs):
            pass

        def model_info(self, repo, revision, files_metadata):
            return SimpleNamespace(sha=repo + '-pinned', siblings=[
                SimpleNamespace(rfilename=name, size=len(content))
                for name, content in repositories[repo].items()])

    def download(repo_id, revision, filename, local_dir, token):
        assert revision == repo_id + '-pinned'
        calls.append((repo_id, filename))
        path = Path(local_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(repositories[repo_id][filename])
        return str(path)

    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(HfApi=Api, hf_hub_download=download))
    monkeypatch.setattr(relay.shutil, 'disk_usage', lambda _: SimpleNamespace(free=10**12))
    (tmp_path / 'configs').mkdir()
    config = tmp_path / 'configs/mage_flow_BooruEssenceFFT.toml'
    config.write_text('must remain untouched')
    settings = dict(stage_root=str(tmp_path), download_mode='standalone', base_repo='base',
                    base_revision='main', checkpoint_repo='custom', checkpoint_revision='main',
                    checkpoint_file=checkpoint, cache_repo='cache', cache_revision='main')
    relay.stage(settings)
    manifest = json.loads((tmp_path / 'manifest.json').read_text())
    paths = {entry['path'] for entry in manifest['files']}
    assert ('base', 'transformer/model.safetensors') not in calls
    assert relay.MODEL + '/vae/model.safetensors' in paths
    assert relay.DATASET + '/cache/mage_flow/latents/0.arrow' in paths
    assert relay.SIDECARS in paths
    assert not any(p.startswith('mnt/') for p in paths)
    assert config.read_text() == 'must remain untouched'
    assert not list((tmp_path / 'payload').rglob('*_nl.txt'))  # archive stays compressed
    assert (tmp_path / 'payload' / Path(manifest['transformer']).parent / 'config.json').read_bytes() == b'{}'
    assert all(item['size'] == (tmp_path / 'payload' / item['path']).stat().st_size for item in manifest['files'])


def test_embedded_worker_matches_tested_source():
    import ast
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / 'DiffusionPipe_Lium_H100.ipynb').read_text(encoding='utf-8'))
    code = ''.join(next(c for c in notebook['cells'] if c.get('id') == 'relay-g1')['source'])
    tree = ast.parse(code)
    embedded = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute) and n.func.attr == 'write_text'
                    and isinstance(n.func.value, ast.Name) and n.func.value.id == 'WORKER_FILE')
    assert ast.literal_eval(embedded.args[0]) == (root / 'utils/notebook_relay.py').read_text(encoding='utf-8')
    compile(relay.REMOTE, '<remote-worker>', 'exec')
