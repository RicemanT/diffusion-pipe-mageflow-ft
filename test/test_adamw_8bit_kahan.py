"""CPU numerical tests with kernel stand-ins; optional real CUDA integration."""
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest
import torch


@pytest.fixture
def module():
    class Base(torch.optim.Optimizer):
        def __init__(self, params, lr=0.01, weight_decay=0.1, **kwargs):
            super().__init__(params, dict(lr=lr, weight_decay=weight_decay,
                betas=(0.9, 0.999), eps=1e-8, max_unorm=0, skip_zeros=False))
            self.non_castable_tensor_keys = set()
            self.optimizer_name = 'adam'

        def init_state(self, group, p, *args):
            self.state[p] = dict(step=0, state1=torch.zeros_like(p, dtype=torch.float32),
                                state2=torch.zeros_like(p, dtype=torch.float32))

        def get_state_buffer(self, p, dtype):
            return torch.zeros_like(p, dtype=dtype)

        def get_config(self, gi, pi, group):
            return group

        def update_step(self, group, p, *args):
            p.mul_(1 - group['lr'] * group['weight_decay'])

    bnb = types.ModuleType('bitsandbytes')
    functional = types.ModuleType('bitsandbytes.functional')
    bnb.optim = types.SimpleNamespace(AdamW8bit=Base)
    bnb.functional = functional

    def kernel32(*a, **kw):
        assert a[12] == 0, 'kernel must never decay compensation'
        assert a[1].dtype == a[2].dtype
        a[2].add_(a[1], alpha=-a[7])

    def kernel8(*a, **kw):
        assert a[16] == 0
        assert a[1].dtype == a[2].dtype
        a[2].add_(a[1], alpha=-a[11])

    def legacy(*a, **kw):
        assert a[16] == 0
        a[2].add_(a[1], alpha=-a[9])

    functional.optimizer_update_32bit = kernel32
    functional.optimizer_update_8bit_blockwise = kernel8
    functional.optimizer_update_8bit = legacy
    path = Path(__file__).resolve().parents[1] / 'optimizers/adamw_8bit.py'
    spec = importlib.util.spec_from_file_location('kahan_under_test', path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'bitsandbytes': bnb, 'bitsandbytes.functional': functional}):
        spec.loader.exec_module(module)
    return module


def setup(module, dtype=torch.bfloat16, **kwargs):
    p = torch.nn.Parameter(torch.ones(32, dtype=dtype))
    p.grad = torch.zeros_like(p)
    opt = module.AdamW8bitKahan([p], **kwargs)
    opt.init_state(opt.param_groups[0], p, 0, 0)
    return p, opt


@pytest.mark.parametrize('mode', ['default', 'sr', 'fp32'])
@pytest.mark.parametrize('kernel', ['32', '8', 'legacy'])
def test_decay_accumulates_below_parameter_ulp(module, mode, kernel):
    dtype = torch.float16 if kernel == 'legacy' else torch.bfloat16
    if mode == 'sr' and kernel == 'legacy':
        pytest.skip('SR is BF16 only')
    p, opt = setup(module, dtype=dtype, lr=0.001, weight_decay=0.1,
                   stochastic_rounding=mode == 'sr', force_kahan_buf_fp32=mode == 'fp32')
    state = opt.state[p]
    if kernel != '32':
        state['state1'] = torch.zeros_like(p, dtype=torch.uint8)
        for key in ('qmap1', 'qmap2', 'absmax1', 'absmax2', 'max1', 'max2', 'new_max1', 'new_max2'):
            state[key] = torch.ones(1)
    if kernel == 'legacy':
        opt.param_groups[0]['block_wise'] = False
    for _ in range(100):
        opt.update_step(opt.param_groups[0], p, 0, 0)
    reconstructed = p.float() + state['shift'].float()
    torch.testing.assert_close(reconstructed, torch.full_like(reconstructed, 0.9999**100), atol=2e-4, rtol=0)
    assert p.max() < 1


def test_residual_preserved_when_switching_precision(module):
    p, opt = setup(module, force_kahan_buf_fp32=True, lr=0, weight_decay=0)
    opt.state[p]['shift'] = torch.full_like(p, 0.001)
    before = p.float() + opt.state[p]['shift'].float()
    opt.update_step(opt.param_groups[0], p, 0, 0)
    assert opt.state[p]['shift'].dtype == torch.float32
    assert 'shift' in opt.non_castable_tensor_keys
    torch.testing.assert_close(p.float() + opt.state[p]['shift'], before)


def test_replicas_ignore_global_rng(module):
    p, opt = setup(module, stochastic_rounding=True)
    q, other = setup(module, stochastic_rounding=True)
    for _ in range(5):
        opt.update_step(opt.param_groups[0], p, 0, 0)
        torch.rand(100)
        other.update_step(other.param_groups[0], q, 0, 0)
    assert torch.equal(p, q)
    assert torch.equal(opt.state[p]['shift'], other.state[q]['shift'])


def test_stochastic_rounding_distribution_and_specials(module):
    generator = torch.Generator().manual_seed(123)
    source = torch.full((100000,), 1 + 1 / 512)
    dest = torch.empty_like(source, dtype=torch.bfloat16)
    module._copy_bf16_stochastic_(dest, source, generator)
    assert set(dest.unique().float().tolist()) == {1.0, 1 + 1 / 128}
    assert abs((dest > 1).float().mean().item() - 0.25) < 0.01
    module._copy_bf16_stochastic_(dest, -source, generator)
    assert abs(dest.float().mean().item() + source[0].item()) < 0.0001
    source = torch.tensor([0., -0., float('inf'), -float('inf'), float('nan')])
    dest = torch.empty_like(source, dtype=torch.bfloat16)
    module._copy_bf16_stochastic_(dest, source, generator)
    torch.testing.assert_close(dest.float(), source, equal_nan=True)


def test_invalid_options_and_fp32_passthrough(module):
    with pytest.raises(ValueError, match='stabilize'):
        setup(module, stabilize=True)
    p, opt = setup(module, dtype=torch.float32)
    assert 'shift' not in opt.state[p]
    opt.update_step(opt.param_groups[0], p, 0, 0)
    torch.testing.assert_close(p, torch.full_like(p, 0.999))
    for key, value in [('max_unorm', 1), ('skip_zeros', True)]:
        p, opt = setup(module)
        opt.param_groups[0][key] = value
        with pytest.raises(ValueError):
            opt.update_step(opt.param_groups[0], p, 0, 0)


@pytest.mark.parametrize('mode', ['default', 'sr', 'fp32'])
@pytest.mark.parametrize('device,size', [('cpu', 32), ('cuda', 32), ('cuda', 8192)])
def test_real_decay_and_resume(mode, device, size):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('requires CUDA bitsandbytes kernels')
    pytest.importorskip('bitsandbytes')
    from optimizers.adamw_8bit import AdamW8bitKahan
    options = dict(lr=0.001, weight_decay=0.1, stochastic_rounding=mode == 'sr',
                   force_kahan_buf_fp32=mode == 'fp32')
    p = torch.nn.Parameter(torch.ones(size, device=device, dtype=torch.bfloat16))
    opt = AdamW8bitKahan([p], **options)
    for _ in range(50):
        p.grad = torch.zeros_like(p)
        opt.step()
    assert p.max() < 1
    torch.testing.assert_close(p.float() + opt.state[p]['shift'].float(),
                               torch.full_like(p.float(), 0.9999**50), atol=2e-4, rtol=0)
    q = torch.nn.Parameter(p.detach().clone())
    restored = AdamW8bitKahan([q], **options)
    restored.load_state_dict(opt.state_dict())
    p.grad = torch.ones_like(p)
    q.grad = p.grad.clone()
    opt.step()
    restored.step()
    torch.testing.assert_close(p, q, rtol=0, atol=0)
    torch.testing.assert_close(opt.state[p]['shift'], restored.state[q]['shift'], rtol=0, atol=0)


def test_real_fp32_buffer_matches_adamw():
    pytest.importorskip('bitsandbytes')
    from optimizers.adamw_8bit import AdamW8bitKahan
    p = torch.nn.Parameter(torch.ones(32, dtype=torch.bfloat16))
    reference = torch.nn.Parameter(p.float())
    options = dict(lr=0.001, weight_decay=0.1, betas=(0.9, 0.999), eps=1e-8)
    opt = AdamW8bitKahan([p], force_kahan_buf_fp32=True, **options)
    ref_opt = torch.optim.AdamW([reference], **options)
    for i in range(20):
        p.grad = torch.full_like(p, (-1)**i * (i + 1) / 20)
        reference.grad = p.grad.float()
        opt.step()
        ref_opt.step()
    torch.testing.assert_close(p.float() + opt.state[p]['shift'], reference,
                               atol=1e-5, rtol=0)
