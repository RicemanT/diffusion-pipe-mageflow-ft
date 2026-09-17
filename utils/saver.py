from pathlib import Path
from contextlib import nullcontext
import os
import shutil
import time
import sys

import torch
import safetensors.torch
from deepspeed import comm as dist
from deepspeed.utils.logging import logger

from utils.common import is_main_process
from utils.distributed_control import broadcast_int


def convert_state_dict_dtype(state_dict, dtype, dtype_for_parameter=None):
    for key, v in state_dict.items():
        target = dtype_for_parameter(key, dtype) if dtype_for_parameter else dtype
        state_dict[key] = v.to(device='cpu', dtype=target)


last_checkpoint_time = None
def need_to_checkpoint(config, epoch=None):
    global last_checkpoint_time

    if epoch is not None:
        if 'checkpoint_every_n_epochs' in config and epoch % config['checkpoint_every_n_epochs'] == 0:
            last_checkpoint_time = time.time()
            return True
        else:
            return False

    if 'checkpoint_every_n_minutes' not in config:
        return False

    checkpoint = False
    # rank 0 tracks if we need to checkpoint, broadcasts to everyone else
    if is_main_process():
        current_time = time.time()
        if last_checkpoint_time is None:
            last_checkpoint_time = current_time
        elif (current_time - last_checkpoint_time) / 60 > config['checkpoint_every_n_minutes']:
            checkpoint = True
            last_checkpoint_time = current_time
    return bool(broadcast_int(checkpoint))


class Saver:
    def __init__(self, args, config, is_adapter, save_root, model, train_dataloader, model_engine, pipeline_model):
        self.args = args
        self.config = config
        self.is_adapter = is_adapter
        self.save_root = Path(save_root)
        self.model = model
        self.train_dataloader = train_dataloader
        self.model_engine = model_engine
        self.pipeline_model = pipeline_model

    def save_adapter(self, name):
        dp_id = self.model_engine.grid.get_data_parallel_rank()
        stage_id = self.model_engine.grid.get_pipe_parallel_rank()
        save_dir = self.save_root / name
        tmp_dir = save_dir / 'tmp'
        if dp_id == 0 and stage_id == 0:
            os.makedirs(tmp_dir, exist_ok=False)
        dist.barrier()
        if dp_id == 0:
            partial_state_dict = {}
            for name, p in self.pipeline_model.named_parameters():
                if p.requires_grad:
                    if not hasattr(p, 'original_name'):
                        logger.warning(f'WARNING: parameter {name} requires_grad but does not have original_name. Not saving it.')
                        continue
                    # TODO: maybe this needs to change if we ever have non-lora adapters?
                    partial_state_dict[p.original_name.replace('.default', '').replace('.modules_to_save', '')] = p.detach()
            if 'save_dtype' in self.config:
                convert_state_dict_dtype(partial_state_dict, self.config['save_dtype'])
            torch.save(partial_state_dict, tmp_dir / f'state_dict_{stage_id}.bin')
        dist.barrier()
        if dp_id == 0 and stage_id == 0:
            state_dict = {}
            for path in tmp_dir.glob('*.bin'):
                state_dict.update(torch.load(path, weights_only=True, map_location='cpu'))
            is_lycoris = bool(getattr(self.model, 'lycoris_modules', None))
            if is_lycoris:
                sd = {'diffusion_model.' + k: v for k, v in state_dict.items()}
                safetensors.torch.save_file(sd, save_dir / 'adapter_model.safetensors', metadata={'format': 'pt'})
            else:
                self.model.save_adapter(save_dir, state_dict)
            shutil.copy(self.args.config, save_dir)
            shutil.rmtree(tmp_dir)

    def save_full_model(self, name):
        dp_id = self.model_engine.grid.get_data_parallel_rank()
        stage_id = self.model_engine.grid.get_pipe_parallel_rank()
        save_dir = self.save_root / name
        tmp_dir = save_dir / 'tmp'
        if dp_id == 0 and stage_id == 0:
            os.makedirs(tmp_dir, exist_ok=False)
        dist.barrier()
        if dp_id == 0:
            # With BF16_Optimizer, we get pickle errors unless we do p.detach(). I have no idea why.
            partial_state_dict = {p.original_name: p.detach() for p in self.pipeline_model.parameters() if hasattr(p, 'original_name')}
            if 'save_dtype' in self.config:
                convert_state_dict_dtype(partial_state_dict, self.config['save_dtype'],
                                         getattr(self.model, 'model_save_dtype', None))
            torch.save(partial_state_dict, tmp_dir / f'state_dict_{stage_id}.bin')
        dist.barrier()
        if dp_id == 0 and stage_id == 0:
            state_dict = {}
            for path in tmp_dir.glob('*.bin'):
                state_dict.update(torch.load(path, map_location='cpu', weights_only=True))
            self.model.save_model(save_dir, state_dict)
            shutil.copy(self.args.config, save_dir)
            shutil.rmtree(tmp_dir)

    def save_model(self, name):
        progress = getattr(self.train_dataloader, 'training_progress', None)
        with progress.phase('saving model') if progress else nullcontext():
            if is_main_process():
                print(f'Saving model to directory {name}')
            if self.is_adapter:
                self.save_adapter(name)
            else:
                self.save_full_model(name)

    def save_checkpoint(self, step, examples):
        progress = getattr(self.train_dataloader, 'training_progress', None)
        with progress.phase('saving checkpoint') if progress else nullcontext():
            self.model_engine.save_checkpoint(
                self.save_root,
                client_state={
                    'step': step,
                    'examples': examples,
                    'custom_loader': self.train_dataloader.state_dict(),
                    'training_plan': getattr(self.train_dataloader, 'training_plan', None),
                    'training_elapsed_seconds': progress.elapsed if progress else 0.0,
                },
                save_latest=True,
                exclude_frozen_parameters=True
            )

    def process_epoch(self, epoch, step, examples):
        checkpointed, saved = False, False
        if self.train_dataloader.epoch != epoch:
            if need_to_checkpoint(self.config, epoch):
                self.save_checkpoint(step, examples)
                checkpointed = True
            if 'save_every_n_epochs' in self.config and epoch % self.config['save_every_n_epochs'] == 0:
                self.save_model(f'epoch{epoch}')
                saved = True
            epoch = self.train_dataloader.epoch
            if epoch > self.config['epochs']:
                return None, checkpointed, saved
            if is_main_process():
                print(f'Started new epoch: {epoch}')
        return epoch, checkpointed, saved

    def process_step(self, step, examples):
        checkpointed, saved = False, False
        # Only rank zero checks signal files. All ranks must make the same
        # checkpoint/exit decision, even with delayed network filesystem visibility.
        signal = 0
        save_signal_file = self.save_root / 'save'
        save_quit_signal_file = self.save_root / 'save_quit'
        if is_main_process():
            if save_quit_signal_file.is_file():
                signal = 2
            elif save_signal_file.is_file():
                signal = 1
        signal = broadcast_int(signal)
        should_manually_save = signal != 0
        should_manually_quit = signal == 2

        if 'save_every_n_steps' in self.config and step % self.config['save_every_n_steps'] == 0:
            self.save_model(f'step{step}')
            saved = True

        if need_to_checkpoint(self.config) or should_manually_save:
            self.save_checkpoint(step, examples)
            checkpointed = True

        # Keep the request on disk until the checkpoint succeeds.
        if should_manually_save and is_main_process():
            signal_file = save_quit_signal_file if should_manually_quit else save_signal_file
            try:
                signal_file.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning(f'Checkpoint saved, but could not remove {signal_file}: {exc}')

        if should_manually_quit:
            print('Manually quitting')
            sys.exit()

        return checkpointed, saved
