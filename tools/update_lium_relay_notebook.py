"""Refresh the self-contained relay cells without rewriting the existing workflow."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / 'DiffusionPipe_Lium_H100.ipynb'


def cell(kind, ident, text):
    result = dict(cell_type=kind, id=ident, metadata={}, source=text.splitlines(keepends=True))
    if kind == 'code':
        result.update(execution_count=None, outputs=[])
    return result


def main():
    notebook = json.loads(PATH.read_text(encoding='utf-8'))
    notebook['cells'] = [c for c in notebook['cells'] if not c.get('id', '').startswith('relay-')]
    intro = '''

### Optional GPU Garden relay (slow pod downloads)
The original direct-download cells **7–9 remain available unchanged**.
Use the **Relay G1–G4 / H1** section at the bottom as an alternative to those downloads.

1. **GPU Garden:** run G1 (setup), G2 (key), then G3 (stage HF files). Run only the relay cells there, not pod setup/training.
2. **H100:** run H1 to authorize the public key from G2 and check the existing `/mnt/configs` files.
3. **GPU Garden:** run G4 to send the staged files over eight SSH connections and validate the H100 destinations.
4. **H100:** run the original setup **1–6**, skip downloads **7–9**, then run **10 (Training)** with `TRUST_CACHE=True`. Output upload **11** stays the same.

No configs are copied, edited or replaced. Existing model/dataset destinations remain `/workspace/Mage-Flow` and `/root/datasets/Booru-Essence-2026`; configs stay on `/mnt/configs`. The training config determines the output location (your current config uses `/workspace/outputs/...`). Do not use **Run All** across both machines.
'''
    # Refresh only our appended guide when rebuilding.
    first = ''.join(notebook['cells'][0]['source']).split('\n### Optional GPU Garden relay')[0]
    notebook['cells'][0]['source'] = (first.rstrip() + '\n' + intro).splitlines(keepends=True)

    setup = '''# GPU GARDEN ONLY: self-contained setup; no H100 setup cells required.
import os, sys, json, subprocess, signal, shutil
from pathlib import Path
from getpass import getpass

STAGE_ROOT = Path.home() / 'lium-training-relay'  # /home/jovyan on GPU Garden
TOOLS_ROOT = Path.home() / '.lium-relay-tools'
TOOLS_ROOT.mkdir(parents=True, exist_ok=True)
STAGE_ROOT.mkdir(parents=True, exist_ok=True)
PACKAGES = TOOLS_ROOT / 'packages'

# Source settings only. H100 destination paths are fixed to the existing notebook.
RELAY_SETTINGS = {
    'stage_root': str(STAGE_ROOT),
    'download_mode': 'standalone',  # 'standalone' or 'full'
    'base_repo': 'mage-flow-community/Mage-Flow',
    'base_revision': 'main',
    'checkpoint_repo': 'RicemanT/MageTrail',
    'checkpoint_revision': 'main',
    'checkpoint_file': 'V0.2/MageTrail-V0.2-Epoch100.safetensors',
    'cache_repo': 'RicemanT/booru-essence-mage-latents',
    'cache_revision': 'main',
    'ssh_host': '93.120.231.186',  # Update if Lium assigns a new pod endpoint.
    'ssh_port': 32280,
    'ssh_key': str(Path.home() / '.ssh/lium_transfer'),
    'ssh_workers': 8,
}

def relay_run(command, env=None):
    process = subprocess.Popen(
        command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, errors='replace', bufsize=1, start_new_session=True,
    )
    try:
        for line in process.stdout:
            print(line, end='', flush=True)
        if process.wait():
            raise RuntimeError('Relay command failed; see the output above.')
    finally:
        # Stop the worker AND its SSH/rsync children on notebook interruption.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        process.stdout.close()

RELAY_ENV = os.environ.copy()
RELAY_ENV.update(PYTHONPATH=str(PACKAGES), PYTHONUNBUFFERED='1',
                 HF_HUB_DISABLE_XET='0', HF_XET_HIGH_PERFORMANCE='1',
                 HF_HUB_DOWNLOAD_TIMEOUT='60', HF_HUB_ETAG_TIMEOUT='30')
RELAY_ENV.pop('HF_HUB_ENABLE_HF_TRANSFER', None)
probe = subprocess.run([sys.executable, '-c', 'import huggingface_hub, hf_xet'],
                       env=RELAY_ENV, capture_output=True)
if probe.returncode:
    relay_run([sys.executable, '-m', 'pip', 'install', '--target', str(PACKAGES),
               '--upgrade', 'huggingface_hub', 'hf-xet'])
if not shutil.which('rsync'):
    prefix = [] if os.geteuid() == 0 else ['sudo', '-n']
    relay_run(prefix + ['apt-get', 'update', '-qq'])
    relay_run(prefix + ['apt-get', 'install', '-y', '-qq', 'rsync'])
token_result = subprocess.run(
    [sys.executable, '-c', "from huggingface_hub import get_token; print(get_token() or '')"],
    env=RELAY_ENV, capture_output=True, text=True, check=True,
)
relay_token = token_result.stdout.strip() or getpass('HF read token (hidden): ').strip()
if not relay_token:
    raise ValueError('An HF read token is required for your cache repository.')
RELAY_ENV['HF_TOKEN'] = relay_token  # Never written to settings or sent to the H100.
SETTINGS_FILE = TOOLS_ROOT / 'settings.json'
SETTINGS_FILE.write_text(json.dumps(RELAY_SETTINGS, indent=2))
WORKER_FILE = TOOLS_ROOT / 'relay_worker.py'
'''
    worker = (ROOT / 'utils/notebook_relay.py').read_text(encoding='utf-8')
    setup += 'WORKER_FILE.write_text(' + repr(worker) + ', encoding="utf-8")\n'
    setup += "print(f'Garden staging: {STAGE_ROOT}; free {shutil.disk_usage(STAGE_ROOT).free/1e9:.1f} GB')\n"
    setup += "print('Setup ready. Run G2 to display your transfer public key, then G3 to stage files.')\n"

    key = '''# GPU GARDEN ONLY: keep the private key here; copy only the printed PUBLIC key.
import subprocess
from pathlib import Path
key = Path(RELAY_SETTINGS['ssh_key']).expanduser()
key.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
if not key.exists():
    subprocess.run(['ssh-keygen', '-q', '-t', 'ed25519', '-N', '',
                    '-C', 'gpu-garden-transfer', '-f', str(key)], check=True)
key.chmod(0o600)
public = subprocess.run(['ssh-keygen', '-y', '-f', str(key)],
                        check=True, capture_output=True, text=True).stdout.strip()
print(public + ' gpu-garden-transfer')
print('Paste the public-key line into H1 on the H100. Existing keys are preserved.')
'''
    auth = '''# H100 ONLY. This cell is independent of the Garden cells and original setup.
import os, subprocess, tempfile
from pathlib import Path
if os.geteuid() != 0:
    raise RuntimeError('Run H1 in the H100 pod root Jupyter session.')
for name in ('mage_flow_BooruEssenceFFT.toml', 'mage_flow_BooruEssenceDataset.toml',
             'MageFlow-Booru-Essence2026-protected-tags.txt'):
    path = Path('/mnt/configs') / name
    if not path.is_file():
        raise FileNotFoundError(f'Attach your original /mnt volume; missing {path}')
public = input('Paste the single PUBLIC key line from Garden G2: ').strip()
if not public.startswith('ssh-ed25519 ') or '\\n' in public:
    raise ValueError('Expected one ssh-ed25519 public-key line.')
with tempfile.NamedTemporaryFile(mode='w', suffix='.pub') as f:
    f.write(public + '\\n')
    f.flush()
    subprocess.run(['ssh-keygen', '-l', '-f', f.name], check=True)
folder = Path('/root/.ssh')
folder.mkdir(mode=0o700, parents=True, exist_ok=True)
folder.chmod(0o700)
authorized = folder / 'authorized_keys'
existing = authorized.read_text() if authorized.exists() else ''
identity = public.split()[:2]
if not any(line.split()[:2] == identity for line in existing.splitlines()):
    with authorized.open('a') as f:
        if existing and not existing.endswith('\\n'):
            f.write('\\n')
        f.write(public + '\\n')
authorized.chmod(0o600)
import shutil
if not shutil.which('rsync'):
    subprocess.run(['apt-get', 'update', '-qq'], check=True)
    subprocess.run(['apt-get', 'install', '-y', '-qq', 'rsync'], check=True)
print('Garden SSH key authorized; rsync ready; existing configs found and unchanged.')
'''
    notebook['cells'].extend([
        cell('markdown', 'relay-guide', '''## Optional relay: GPU Garden → H100

Run cells only on the machine named in each heading. This section replaces direct downloads **7–9**, not trainer installation, CUDA setup, training or output uploads.

Garden needs space for the supporting model, custom checkpoint, cached latents and compressed sidecars. G3 pins repository revisions, checks space, and reuses HF downloads on reruns. The raw image dataset is unnecessary for this prebuilt cache. H100 configs on `/mnt/configs` are read and validated, never transferred or rewritten.

G4 uses eight independent rsync connections with balanced file lists, checksum comparison, retained partial files, and local per-connection logs. Parallelism is across files: a single large file uses one connection, so speed can fall near the end. No `--delete` is used. Source files remain on Garden after transfer. Sidecars are extracted only on the H100. Missing config files or incompatible paths fail explicitly instead of silently rewriting your settings.
'''),
        cell('markdown', 'relay-g1-title', '### Relay G1 — GPU Garden: setup and source settings\n'),
        cell('code', 'relay-g1', setup),
        cell('markdown', 'relay-g2-title', '### Relay G2 — GPU Garden: SSH public key\n'),
        cell('code', 'relay-g2', key),
        cell('markdown', 'relay-h1-title', '### Relay H1 — H100: authorize Garden and check existing configs\nRun on the H100 using its copy of this notebook. No Garden variables are required.\n'),
        cell('code', 'relay-h1', auth),
        cell('markdown', 'relay-g3-title', '### Relay G3 — GPU Garden: download required HF files\nRun G1 first. This can run before renting the full H100 pod.\n'),
        cell('code', 'relay-g3', "SETTINGS_FILE.write_text(json.dumps(RELAY_SETTINGS, indent=2))\nrelay_run([sys.executable, '-u', str(WORKER_FILE), 'stage', str(SETTINGS_FILE)], env=RELAY_ENV)\n"),
        cell('markdown', 'relay-g4-title', '### Relay G4 — GPU Garden: transfer to H100 and validate\nRun H1 on the destination first. Check the SSH host/port in G1 if the pod changed. Interrupting retains partial files; rerun this cell to resume. Wait for `TRANSFER VERIFIED` before training.\n'),
        cell('code', 'relay-g4', "SETTINGS_FILE.write_text(json.dumps(RELAY_SETTINGS, indent=2))\nrelay_run([sys.executable, '-u', str(WORKER_FILE), 'transfer', str(SETTINGS_FILE)], env=RELAY_ENV)\n"),
    ])
    for c in notebook['cells']:
        if c['cell_type'] == 'code':
            c.update(execution_count=None, outputs=[])
    PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
