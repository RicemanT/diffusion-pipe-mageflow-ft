"""Portable worker embedded in the Lium notebook; no trainer imports required."""
import argparse
import concurrent.futures
import fnmatch
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import queue
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time


WEIGHT_PATTERNS = [f'transformer/*{suffix}' for suffix in (
    '.safetensors', '.bin', '.pt', '.pth', '.ckpt', '.msgpack', '.h5',
    '.onnx', '.onnx_data', '.index.json')]
DATASET = 'root/datasets/Booru-Essence-2026'
MODEL = 'workspace/Mage-Flow'
SIDECARS = 'workspace/.mageflow-relay/sidecars.tar.gz'


def safe_relative(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or '..' in path.parts or '\\' in name:
        raise ValueError(f'Unsafe relative path: {name!r}')
    return path


def selected_base(name, mode):
    return mode == 'full' or not any(fnmatch.fnmatchcase(name, p) for p in WEIGHT_PATTERNS)


def checkpoint_directory(repo, revision, filename):
    safe_relative(filename)
    identity = hashlib.sha256(f'{repo}@{revision}:{filename}'.encode()).hexdigest()[:12]
    return f'{MODEL}/custom_transformers/{identity}'


def sidecar_members(archive):
    members = archive.getmembers()
    count = nl = 0
    for member in members:
        safe_relative(member.name)
        if not (member.isfile() or member.isdir()):
            raise ValueError(f'Unexpected sidecar archive entry: {member.name}')
        if member.isfile() and member.name.endswith('.txt'):
            count += 1
            nl += member.name.endswith('_nl.txt')
    if count <= 1000 or not nl:
        raise ValueError(f'Incomplete sidecars: {count} captions, {nl} NL captions')
    return members, count, nl


def balanced_lanes(entries, workers):
    lanes = [[] for _ in range(min(workers, len(entries)))]
    loads = [0] * len(lanes)
    for entry in sorted(entries, key=lambda e: e['size'], reverse=True):
        index = min(range(len(lanes)), key=lambda i: loads[i])
        lanes[index].append(entry['path'])
        loads[index] += entry['size']
    return lanes


def stage(settings):
    from huggingface_hub import HfApi, hf_hub_download
    root = Path(settings['stage_root']).expanduser().resolve()
    payload = root / 'payload'
    payload.mkdir(parents=True, exist_ok=True)
    marker = root / 'manifest.json'
    # A failed restaging must not leave a stale "ready" manifest.
    marker.unlink(missing_ok=True)
    mode = settings['download_mode']
    if mode not in ('standalone', 'full'):
        raise ValueError('download_mode must be standalone or full')
    api = HfApi(token=os.environ.get('HF_TOKEN') or False)
    jobs = []
    revisions = {}

    def add_repo(repo, revision, prefix, select):
        info = api.model_info(repo, revision=revision, files_metadata=True)
        revisions[f'{repo}@{revision}'] = info.sha
        for item in info.siblings:
            safe_relative(item.rfilename)
            if select(item.rfilename):
                if item.size is None:
                    raise RuntimeError(f'Unknown download size: {item.rfilename}')
                jobs.append(dict(repo=repo, revision=info.sha, filename=item.rfilename,
                                 prefix=prefix, size=item.size))

    add_repo(settings['base_repo'], settings['base_revision'], MODEL,
             lambda name: selected_base(name, mode))
    transformer = None
    if mode == 'standalone':
        repo, revision, filename = (settings[k] for k in
                                    ('checkpoint_repo', 'checkpoint_revision', 'checkpoint_file'))
        prefix = checkpoint_directory(repo, revision, filename)
        add_repo(repo, revision, prefix, lambda name: name == filename)
        transformer = f'{prefix}/{filename}'
        if not any(j['prefix'] == prefix for j in jobs):
            raise RuntimeError(f'Checkpoint not found: {repo}/{filename}')
    cache_prefix = f'{DATASET}/cache/mage_flow'
    add_repo(settings['cache_repo'], settings['cache_revision'], cache_prefix,
             lambda name: name != 'sidecars.tar.gz')
    add_repo(settings['cache_repo'], revisions[f"{settings['cache_repo']}@{settings['cache_revision']}"],
             'workspace/.mageflow-relay', lambda name: name == 'sidecars.tar.gz')
    if not any(j['filename'] == 'sidecars.tar.gz' for j in jobs):
        raise RuntimeError('Cache repository has no sidecars.tar.gz')

    missing_bytes = 0
    for job in jobs:
        path = payload / job['prefix'] / job['filename']
        if not path.is_file() or path.stat().st_size != job['size']:
            missing_bytes += job['size']
    free = shutil.disk_usage(root).free
    print(f'Pinned downloads: {len(jobs)} files; {sum(j["size"] for j in jobs)/1e9:.2f} GB total', flush=True)
    print(f'Estimated additional space: {missing_bytes/1e9:.2f} GB; free: {free/1e9:.2f} GB', flush=True)
    if free < missing_bytes + 2 * 1024**3:
        raise RuntimeError('Insufficient GPU Garden storage (includes 2 GiB reserve).')

    def download(job):
        path = hf_hub_download(repo_id=job['repo'], revision=job['revision'],
                               filename=job['filename'], local_dir=payload / job['prefix'],
                               token=os.environ.get('HF_TOKEN') or False)
        if Path(path).stat().st_size != job['size']:
            raise RuntimeError(f'Incomplete download: {path}')
        return {'path': f"{job['prefix']}/{job['filename']}", 'size': job['size']}

    entries = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for future in concurrent.futures.as_completed([pool.submit(download, j) for j in jobs]):
            entries.append(future.result())
            print(f'Ready {len(entries)}/{len(jobs)}: {entries[-1]["path"]}', flush=True)
    if transformer:
        config = payload / str(PurePosixPath(transformer).parent / 'config.json')
        shutil.copy2(payload / MODEL / 'transformer/config.json', config)
        entries.append({'path': config.relative_to(payload).as_posix(), 'size': config.stat().st_size})
    for name in ('model_index.json', 'transformer/config.json', 'vae/config.json', 'text_encoder/config.json'):
        if not (payload / MODEL / name).is_file():
            raise RuntimeError('Required model file missing: ' + name)
    with tarfile.open(payload / SIDECARS, 'r:gz') as archive:
        _, count, nl = sidecar_members(archive)
        print(f'Sidecars checked: {count} captions ({nl} NL); extraction happens on H100.', flush=True)
    cache = payload / cache_prefix
    if not (cache / 'metadata/grouping_keys.json').is_file() or not list(cache.glob('metadata/grouped_metadata_*')):
        raise RuntimeError('Required cached metadata is missing.')
    manifest = dict(version=1, files=entries, revisions=revisions, transformer=transformer,
                    dataset=DATASET, model=MODEL, sidecars=SIDECARS)
    marker.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Garden staging complete: {marker}', flush=True)
    print('No original images required for this existing --trust_cache workflow.', flush=True)


# Executed on the H100 using only its Python standard library.
REMOTE = r'''
import json, os, sys, tarfile, shutil, tomllib
from pathlib import Path, PurePosixPath
manifest = json.load(sys.stdin)
def target(name):
    rel = PurePosixPath(name)
    if rel.is_absolute() or '..' in rel.parts or '\\' in name:
        raise ValueError('Unsafe manifest path')
    if not name.startswith(('workspace/Mage-Flow/', 'workspace/.mageflow-relay/',
                            'root/datasets/Booru-Essence-2026/')):
        raise ValueError('Unexpected destination: ' + name)
    return Path('/') / name
phase = sys.argv[1]
if phase == 'preflight':
    if not shutil.which('rsync'):
        raise RuntimeError('Install rsync on the H100 first: apt-get update && apt-get install -y rsync')
    devices = {}
    for item in manifest['files']:
        path = target(item['path'])
        path.parent.mkdir(parents=True, exist_ok=True)
        device = path.parent.stat().st_dev
        row = devices.setdefault(device, [path.parent, 0])
        if not path.is_file() or path.stat().st_size != item['size']:
            row[1] += item['size']
    # Conservatively budget all remaining files, plus space for sidecars and rsync temp files.
    reserve = max((i['size'] for i in manifest['files']), default=0) + 2 * 1024**3
    for path, needed in devices.values():
        free = shutil.disk_usage(path).free
        print(f'{path}: free {free/1e9:.2f} GB; estimated new files {needed/1e9:.2f} GB', flush=True)
        if free < needed + reserve:
            raise RuntimeError('Insufficient destination free space including temporary-file reserve')
    for name in ('mage_flow_BooruEssenceFFT.toml', 'mage_flow_BooruEssenceDataset.toml',
                 'MageFlow-Booru-Essence2026-protected-tags.txt'):
        rel = 'mnt/configs/' + name
        if not Path('/' + rel).is_file():
            raise RuntimeError('Missing existing volume config: /' + rel + '; attach the original /mnt volume')
    cfg = tomllib.loads(Path('/mnt/configs/mage_flow_BooruEssenceFFT.toml').read_text())
    if cfg['model']['diffusers_path'] != '/' + manifest['model']:
        raise RuntimeError('Existing diffusers_path differs from relay destination; no config was changed')
    if manifest['transformer'] and cfg['model'].get('transformer_path') != '/' + manifest['transformer']:
        raise RuntimeError('Existing transformer_path differs from selected checkpoint; check Garden source settings')
    if cfg['dataset'] != '/mnt/configs/mage_flow_BooruEssenceDataset.toml':
        raise RuntimeError('Existing dataset config path differs from this notebook; no config was changed')
    print('SSH, rsync, config presence and destination space checked.', flush=True)
else:
    for item in manifest['files']:
        path = target(item['path'])
        if not path.is_file() or path.stat().st_size != item['size']:
            raise RuntimeError('Missing/incomplete transfer: ' + str(path))
    dataset = Path('/') / manifest['dataset']
    with tarfile.open(target(manifest['sidecars']), 'r:gz') as archive:
        members = archive.getmembers()
        for member in members:
            rel = PurePosixPath(member.name)
            if rel.is_absolute() or '..' in rel.parts or not (member.isfile() or member.isdir()):
                raise RuntimeError('Unsafe sidecar archive member')
        archive.extractall(dataset, members=members, filter='data')
    captions = list(dataset.rglob('*.txt'))
    if len(captions) <= 1000 or not any(p.name.endswith('_nl.txt') for p in captions):
        raise RuntimeError('Incomplete extracted captions')
    (dataset / '.sidecars_complete').write_text(str(len(captions)) + '\n')
    cfg_path = Path('/mnt/configs/mage_flow_BooruEssenceFFT.toml')
    cfg = tomllib.loads(cfg_path.read_text())
    expected_model = '/' + manifest['model']
    if cfg['model']['diffusers_path'] != expected_model:
        raise RuntimeError('Config diffusers_path does not match relay destination')
    if manifest['transformer'] and cfg['model'].get('transformer_path') != '/' + manifest['transformer']:
        raise RuntimeError('Config transformer_path does not match selected checkpoint: /' + manifest['transformer'])
    dataset_config = Path(cfg['dataset'])
    data_cfg = tomllib.loads(dataset_config.read_text())
    def paths(value):
        if isinstance(value, dict):
            for k, v in value.items():
                if k == 'path' and isinstance(v, str):
                    yield v
                else:
                    yield from paths(v)
        elif isinstance(value, list):
            for v in value:
                yield from paths(v)
    dataset_paths = list(paths(data_cfg))
    if not dataset_paths or any(p != str(dataset) for p in dataset_paths):
        raise RuntimeError('Dataset paths must match cached path exactly: ' + str(dataset))
    protected = cfg['model'].get('protected_tags_file')
    if protected and not Path(protected).is_file():
        raise RuntimeError('Missing protected tags file: ' + protected)
    cache = dataset / 'cache/mage_flow'
    if not (cache / 'metadata/grouping_keys.json').is_file() or not list(cache.glob('metadata/grouped_metadata_*')):
        raise RuntimeError('Missing latent-cache metadata')
    Path(cfg['output_dir']).mkdir(parents=True, exist_ok=True)
    print('TRANSFER VERIFIED: model, checkpoint, cache, sidecars and config paths ready.', flush=True)
    print('Output path preserved: ' + cfg['output_dir'], flush=True)
    print('Run H100 setup 1-6, then Training (10) with TRUST_CACHE=True; skip downloads 7-9.', flush=True)
'''


def transfer(settings):
    root = Path(settings['stage_root']).expanduser().resolve()
    payload = root / 'payload'
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    workers = int(settings.get('ssh_workers', 8))
    if not 1 <= workers <= 32:
        raise ValueError('ssh_workers must be between 1 and 32')
    if not shutil.which('rsync'):
        raise RuntimeError('Install rsync on Garden first: sudo apt-get update && sudo apt-get install -y rsync')
    key = Path(settings['ssh_key']).expanduser()
    if not key.is_file():
        raise RuntimeError('Run the Garden SSH key cell first.')
    ssh = ['ssh', '-T', '-i', str(key), '-p', str(settings['ssh_port']),
           '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes',
           '-o', 'StrictHostKeyChecking=accept-new', '-o', 'Compression=no',
           '-o', 'ControlMaster=no', '-o', 'ControlPath=none',
           '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=10',
           '-o', 'ServerAliveCountMax=3']
    host = settings['ssh_host']
    if not re.fullmatch(r'[A-Za-z0-9.-]+', host):
        raise ValueError('Enter an IP address or hostname only, without ssh/user/port.')
    destination = 'root@' + host
    for item in manifest['files']:
        safe_relative(item['path'])
        path = payload / item['path']
        if not path.is_file() or path.stat().st_size != item['size']:
            raise RuntimeError('Garden staging file missing/changed: ' + str(path))

    def remote(phase):
        result = subprocess.run(ssh + [destination, 'python3 -c ' + shlex.quote(REMOTE) + ' ' + phase],
                                input=json.dumps(manifest), text=True, capture_output=True)
        print(result.stdout, flush=True)
        if result.returncode:
            raise RuntimeError(result.stderr)

    remote('preflight')
    lanes = balanced_lanes(manifest['files'], workers)
    processes = []
    updates = queue.Queue()
    totals = [0] * len(lanes)
    logs = root / 'transfer-logs'
    logs.mkdir(exist_ok=True)

    def read_progress(index, process):
        with (logs / f'lane-{index+1}.log').open('w', encoding='utf-8') as log:
            for line in process.stdout:
                log.write(line)
                log.flush()
                match = re.match(r'\s*([\d,]+)\s+\d+%', line)
                if match:
                    updates.put((index, int(match[1].replace(',', ''))))

    started = time.monotonic()
    print(f'Starting {len(lanes)} rsync connections. Partial transfers are retained for reruns.', flush=True)
    print('Speed measures rsync processed bytes; reused/delta data can inflate it on reruns.', flush=True)
    with tempfile.TemporaryDirectory(prefix='rsync-lists-', dir=root) as folder:
        readers = []
        try:
            for index, names in enumerate(lanes):
                listing = Path(folder) / str(index)
                listing.write_bytes(b'\0'.join(n.encode() for n in names) + b'\0')
                command = ['rsync', '-rt', '--checksum', '--relative', '--from0',
                           '--files-from=' + str(listing), '--partial-dir=.rsync-partial',
                           '--info=progress2', '--outbuf=L', '-e', shlex.join(ssh),
                           str(payload) + '/', destination + ':/']
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True, errors='replace', bufsize=1)
                processes.append(process)
                reader = threading.Thread(target=read_progress, args=(index, process), daemon=True)
                reader.start()
                readers.append(reader)
            previous_time, previous_total = started, 0
            while any(p.poll() is None for p in processes):
                time.sleep(1)
                while not updates.empty():
                    index, value = updates.get_nowait()
                    totals[index] = value
                now, total = time.monotonic(), sum(totals)
                print(f'Elapsed {now-started:.0f}s | processed {total/1e9:.2f} GB | '
                      f'{max(0,total-previous_total)/(now-previous_time)/1e6:.2f} MB/s | '
                      f'active {sum(p.poll() is None for p in processes)}/{len(lanes)}', flush=True)
                previous_time, previous_total = now, total
                if any(p.poll() not in (None, 0) for p in processes):
                    raise RuntimeError(f'rsync failed; inspect {logs}. Rerun to resume.')
            if any(p.returncode for p in processes):
                raise RuntimeError(f'rsync failed; inspect {logs}')
        finally:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
            for process in processes:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            for reader in readers:
                reader.join(timeout=5)
            for process in processes:
                process.stdout.close()
    print('All rsync connections completed; extracting captions and validating training paths...', flush=True)
    remote('finalize')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['stage', 'transfer'])
    parser.add_argument('settings')
    args = parser.parse_args()
    settings = json.loads(Path(args.settings).read_text(encoding='utf-8'))
    (stage if args.action == 'stage' else transfer)(settings)
