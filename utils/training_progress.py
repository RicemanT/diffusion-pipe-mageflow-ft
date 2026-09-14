"""Rank-zero terminal progress with measured optimizer-step throughput."""
from collections import deque
from contextlib import ExitStack, contextmanager, redirect_stdout
import logging
import sys
import time

from tqdm import tqdm
from tqdm.contrib import DummyTqdmFile
from tqdm.contrib.logging import logging_redirect_tqdm


def duration(seconds):
    if seconds is None:
        return '--:--:--'
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


def batch_samples(dataset, batch_index):
    """Global sample slots in this actual bucket batch, including its padding."""
    bucket_index, _ = dataset.iteration_order[batch_index]
    return dataset.buckets[bucket_index].global_batch_size


class TrainingProgress:
    def __init__(self, total, epochs, *, initial=0, elapsed=0.0, enabled=True,
                 main_process=True, image_only=True, epoch=1, file=None, clock=time.perf_counter):
        if enabled not in (True, False, 'always'):
            raise ValueError('progress_bar must be true, false, or "always".')
        self.file = file if file is not None else sys.stderr
        self.live = main_process and enabled is not False and (
            enabled == 'always' or self.file.isatty())
        self.total, self.epochs, self.initial, self.completed = total, epochs, initial, initial
        self.prior_elapsed = max(0.0, elapsed)
        self.clock = clock
        self.started = clock()
        self.window = deque(maxlen=50)
        self.sample_unit = 'img' if image_only else 'sample'
        self.bar = self.status = None
        self.epoch = epoch
        self.details = ''
        self.context = ExitStack()

    @property
    def elapsed(self):
        return self.prior_elapsed + self.clock() - self.started

    def __enter__(self):
        self.started = self.clock()
        if self.live:
            # Redirect ordinary prints and Python console handlers, but leave file
            # handlers and stderr available for nested validation tqdm bars.
            self.context.enter_context(logging_redirect_tqdm(
                loggers=[logging.getLogger(), logging.getLogger('DeepSpeed')]))
            self.context.enter_context(redirect_stdout(DummyTqdmFile(sys.stdout)))
            self.bar = tqdm(total=self.total, initial=self.initial, desc=f'Epoch {self.epoch}/{self.epochs}',
                            file=self.file, position=0, dynamic_ncols=True, mininterval=0.25,
                            bar_format='{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{postfix}]')
            self.status = tqdm(total=0, file=self.file, position=1, dynamic_ncols=True,
                               bar_format='{desc}', leave=False)
            self._refresh_time()
        return self

    def metrics(self):
        seconds = sum(t for t, _ in self.window)
        samples = sum(n for _, n in self.window)
        steps = len(self.window)
        session_steps = self.completed - self.initial
        session_elapsed = max(0.0, self.clock() - self.started)
        eta = session_elapsed / session_steps * (self.total - self.completed) if session_steps else None
        return {
            'train/seconds_per_step': seconds / steps if steps else 0.0,
            'train/steps_per_second': steps / seconds if seconds else 0.0,
            'train/samples_per_second': samples / seconds if seconds else 0.0,
            'train/elapsed_seconds': self.elapsed,
            'train/eta_seconds': eta,
        }

    def update(self, step, epoch, epoch_step, steps_per_epoch, *, loss, lrs, samples, seconds):
        if step != self.completed + 1 or step > self.total:
            raise ValueError(f'Progress expected step {self.completed + 1} of {self.total}, got {step}.')
        self.completed, self.epoch = step, epoch
        self.window.append((max(seconds, 1e-9), samples))
        stats = self.metrics()
        if lrs:
            low, high = min(lrs), max(lrs)
            lr = f'{low:.2g}' if low == high else f'{low:.2g}..{high:.2g}'
        else:
            lr = '?'
        self.details = (f'ep {epoch_step}/{steps_per_epoch} | loss {loss:.4f} | lr {lr} | '
                        f"{stats['train/seconds_per_step']:.2f}s/step "
                        f"{stats['train/steps_per_second']:.2f}step/s | "
                        f"{stats['train/samples_per_second']:.1f}{self.sample_unit}/s")
        if self.live:
            self.bar.set_description_str(f'Epoch {epoch}/{self.epochs}', refresh=False)
            self.bar.set_postfix_str(f"elapsed {duration(self.elapsed)} | ETA {duration(stats['train/eta_seconds'])}", refresh=False)
            self.status.set_description_str(self.details, refresh=False)
            rendered = self.bar.update(step - self.bar.n)
            if rendered or step == self.initial + 1 or step == self.total:
                if not rendered:
                    self.bar.refresh()
                self.status.refresh()
        return stats

    @contextmanager
    def phase(self, name):
        if self.live:
            self.bar.set_description_str(f'Epoch {self.epoch}/{self.epochs} {name}')
        try:
            yield
        finally:
            if self.live:
                self._refresh_time()
                self.bar.set_description_str(f'Epoch {self.epoch}/{self.epochs}')

    def _refresh_time(self):
        stats = self.metrics()
        self.bar.set_postfix_str(
            f"elapsed {duration(stats['train/elapsed_seconds'])} | ETA {duration(stats['train/eta_seconds'])}",
            refresh=False,
        )

    def __exit__(self, exc_type, exc, traceback):
        try:
            if self.live:
                self._refresh_time()
                if exc_type is not None:
                    self.bar.set_description_str(f'Stopped at epoch {self.epoch}/{self.epochs}')
                self.status.close()
                self.bar.close()
        finally:
            self.context.close()
