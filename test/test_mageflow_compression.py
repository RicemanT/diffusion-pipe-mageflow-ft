"""Numerical compressed-model regressions without pretrained weights or CUDA."""

import ast
import copy
import json
import os
import sys
from typing import Any
from pathlib import Path
from types import SimpleNamespace

import pytest
import safetensors
from safetensors.torch import load_file, save_file
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from test.test_mageflow_training import load_training_components, tiny_block
from test.test_mageflow_execution import vendored_block
from utils.mageflow_compression import (
    ARCHITECTURE, ModulationProjection, compressed_parameter,
    install_compressed_modulation, read_transformer_config, resolve_modulation_dtype,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module', autouse=True)
def small_tensor_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class Rope(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.frequencies = torch.ones(4, 4, dtype=torch.complex64)

    def forward(self, shapes, device):
        _, h, w = shapes[0]
        return torch.ones(h * w, 4, device=device, dtype=torch.complex64)


class Time(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 16)

    def forward(self, t, img):
        return self.linear(t[:, None])


def model_fixture():
    torch.manual_seed(61)
    model = nn.Module()
    model.inner_dim = 16
    model.img_in = nn.Linear(4, 16)
    model.txt_norm = nn.LayerNorm(8)
    model.txt_in = nn.Linear(8, 16)
    model.time_text_embed = Time()
    model.pos_embed = Rope()
    model.transformer_blocks = nn.ModuleList([tiny_block().float(), tiny_block().float()])
    model.norm_out = nn.Module()
    model.norm_out.linear = nn.Linear(16, 32)
    model.norm_out.silu = nn.SiLU()
    model.norm_out.norm = nn.LayerNorm(16, elementwise_affine=False)
    model.proj_out = nn.Linear(16, 4)
    install_compressed_modulation(model, 4)
    for module in model.modules():
        if isinstance(module, ModulationProjection):
            nn.init.normal_(module.weight, std=0.05)
            nn.init.normal_(module.bias, std=0.02)
    return model


def layers_for(model, backend='sdpa'):
    ns = load_training_components()
    offloader = SimpleNamespace(wait_for_block=lambda i: None, submit_move_blocks_forward=lambda i: None)
    return [ns['InitialLayer'](model, attention_backend=backend), *[
        ns['TransformerLayer'](b, i, 2, offloader, attention_backend=backend)
        for i, b in enumerate(model.transformer_blocks)
    ], ns['FinalLayer'](model)]


def inputs(dtype=torch.float32):
    return (torch.randn(2, 4, 2, 2).to(dtype), torch.randn(2, 3, 8).to(dtype),
            torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool),
            torch.tensor([0.2, 0.7]), torch.tensor([[2, 2], [2, 2]]))


@pytest.mark.parametrize('reentrant', [None, False, True])
@pytest.mark.parametrize('backend', ['sdpa', 'reference'])
def test_pipeline_matches_expanded_dense_model_and_shared_gradients(reentrant, backend):
    model = model_fixture()
    dense = copy.deepcopy(model)
    down = dense.modulation_down
    for block in dense.transformer_blocks:
        block.compressed_modulation = False
        for stream in ('img_mod', 'txt_mod'):
            head = getattr(block, stream)[1]
            linear = nn.Linear(16, 96)
            with torch.no_grad():
                linear.weight.copy_(head.weight @ down.weight)
                linear.bias.copy_(head.weight @ down.bias + head.bias)
            setattr(block, stream, nn.Sequential(nn.SiLU(), linear))
    del dense.modulation_down
    batch = inputs()

    def run(m, checkpointed):
        layers = layers_for(m, backend)
        state = layers[0](batch)
        original_temb = state[3]
        for layer in layers[1:-1]:
            state = layer(state) if checkpointed is None else checkpoint(
                layer, state, use_reentrant=False) if not checkpointed else checkpoint(
                    lambda *xs, layer=layer: layer(xs), *state, use_reentrant=True)
        torch.testing.assert_close(state[3], original_temb, rtol=0, atol=0)
        return layers[-1](state)

    calls = []
    handle = model.modulation_down.register_forward_hook(lambda *args: calls.append(1))
    actual = run(model, reentrant)
    expected = run(dense, None)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    actual.square().mean().backward()
    expected.square().mean().backward()
    handle.remove()
    assert len(calls) == 1  # no repeated/shared weight registration in blocks
    for name, p in model.named_parameters():
        if compressed_parameter(name):
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
            assert p.grad.dtype == torch.float32
    torch.testing.assert_close(model.time_text_embed.linear.weight.grad,
                               dense.time_text_embed.linear.weight.grad, rtol=2e-5, atol=2e-6)
    # Shared-factor gradient must agree with a separate eager execution too.
    eager = copy.deepcopy(model)
    eager.zero_grad(set_to_none=True)
    run(eager, None).square().mean().backward()
    torch.testing.assert_close(model.modulation_down.weight.grad, eager.modulation_down.weight.grad)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_autocast_device_moves_and_small_optimizer_updates(dtype):
    model = model_fixture()
    projection = model.modulation_down
    projection.parameter_dtype = dtype
    projection.to(dtype=dtype)
    original = projection.weight.detach().clone()
    model.bfloat16()
    assert model.img_in.weight.dtype == torch.bfloat16
    assert projection.weight.dtype == dtype
    torch.testing.assert_close(projection.weight, original, rtol=0, atol=0)
    optimizer = torch.optim.AdamW(projection.parameters(), lr=1e-5)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        output = projection(torch.randn(2, 16, dtype=torch.bfloat16))
        assert output.dtype == dtype
        output.float().square().mean().backward()
    assert projection.weight.grad.dtype == dtype
    optimizer.step()
    if dtype == torch.float32:
        assert not torch.equal(original, projection.weight)


def test_complete_mixed_dtype_pipeline_backward():
    model = model_fixture().bfloat16()
    layers = layers_for(model)
    state = inputs(torch.bfloat16)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        for layer in layers:
            state = layer(state)
        assert state.dtype == torch.bfloat16
        state.float().square().mean().backward()
    for name, parameter in model.named_parameters():
        expected_dtype = torch.float32 if compressed_parameter(name) else torch.bfloat16
        assert parameter.dtype == expected_dtype
        if parameter.grad is not None:
            assert parameter.grad.dtype == expected_dtype
            assert parameter.grad.isfinite().all()


def pipeline_methods():
    tree = ast.parse((ROOT / 'models/mage_flow.py').read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MageFlowPipeline')
    cls.bases = []
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef)
                and n.name in ('save_model', 'model_save_dtype', 'training_communication_dtype',
                               '_transformer_params', 'load_diffusion_model')]
    ns = dict(torch=torch, safetensors=safetensors, json=json,
              ARCHITECTURE=ARCHITECTURE, compressed_parameter=compressed_parameter,
              Path=Path, read_transformer_config=read_transformer_config,
              resolve_modulation_dtype=resolve_modulation_dtype,
              install_compressed_modulation=install_compressed_modulation,
              MageFlowParams=SimpleNamespace, MageFlow=lambda params: model_fixture(),
              validate_execution_config=lambda *args, **kwargs: None,
              load_file=load_file, KEEP_IN_HIGH_PRECISION=[])
    exec(compile(ast.Module(body=[cls], type_ignores=[]), '<pipeline>', 'exec'), ns)
    return ns['MageFlowPipeline']()


def test_pipeline_loader_materializes_meta_weights_and_unregistered_rope(tmp_path):
    model = model_fixture()
    config = dict(in_channels=4, out_channels=4, context_in_dim=8, hidden_size=16,
                  num_heads=2, depth=2, axes_dim=[2, 2, 4], modulation_rank=4)
    path = tmp_path / 'model.safetensors'
    save_file(model.state_dict(), path, metadata={
        'architecture': ARCHITECTURE, 'model_config': json.dumps(config)})
    owner = pipeline_methods()
    owner.model_config = {'transformer_path': str(path), 'dtype': torch.bfloat16}
    owner.config = {'model': owner.model_config}
    owner.load_diffusion_model()
    assert owner.transformer.pos_embed.frequencies.device.type == 'cpu'
    for name, p in owner.transformer.named_parameters():
        assert p.device.type == 'cpu'
        assert p.original_name == name
        assert p.dtype == (torch.float32 if compressed_parameter(name) else torch.bfloat16)
    broken = model.state_dict()
    del broken['transformer_blocks.0.img_mod.1.bias']
    save_file(broken, path, metadata={
        'architecture': ARCHITECTURE, 'model_config': json.dumps(config)})
    with pytest.raises(RuntimeError, match='Missing key'):
        owner.load_diffusion_model()


def actual_transformer_class(monkeypatch):
    block = vendored_block(monkeypatch)
    modules = sys.modules[type(block).__module__]
    tree = ast.parse((ROOT / 'Mage/mage_flow/models/mage_flow.py').read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MageFlow')
    ns = dict(vars(modules), MageFlowParams=SimpleNamespace, Any=Any)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), '<actual_mageflow>', 'exec'), ns)
    return ns['MageFlow']


def test_actual_architecture_meta_loading_rope_and_forward(monkeypatch, tmp_path):
    factory = actual_transformer_class(monkeypatch)
    config = dict(in_channels=4, out_channels=4, context_in_dim=8, hidden_size=16,
                  num_heads=2, depth=2, axes_dim=[2, 2, 4], checkpoint=False, patch_size=1)
    model = factory(SimpleNamespace(**config))
    install_compressed_modulation(model, 4)
    for module in model.modules():
        if isinstance(module, ModulationProjection):
            nn.init.normal_(module.weight, std=0.05)
            nn.init.zeros_(module.bias)
    path = tmp_path / 'model.safetensors'
    save_file(model.state_dict(), path, metadata={
        'architecture': ARCHITECTURE,
        'model_config': json.dumps(dict(config, modulation_rank=4))})
    owner = pipeline_methods()
    owner.load_diffusion_model.__globals__['MageFlow'] = factory
    owner.model_config = {'transformer_path': str(path), 'dtype': torch.bfloat16}
    owner.config = {'model': owner.model_config}
    owner.load_diffusion_model()
    assert owner.transformer.pos_embed.pos_freqs.device.type == 'cpu'
    assert owner.transformer.pos_embed.neg_freqs.device.type == 'cpu'
    state = inputs(torch.bfloat16)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        for layer in layers_for(owner.transformer):
            state = layer(state)
        assert state.isfinite().all()
        state.float().square().mean().backward()
    assert owner.transformer.modulation_down.weight.grad.isfinite().all()


def test_published_rank256_architecture_parameter_count(monkeypatch):
    factory = actual_transformer_class(monkeypatch)
    params = SimpleNamespace(in_channels=128, out_channels=128, context_in_dim=2560,
                             hidden_size=3072, num_heads=24, depth=12,
                             axes_dim=[16, 56, 56], checkpoint=False, patch_size=1)
    with torch.device('meta'):
        model = factory(params)
        install_compressed_modulation(model, 256)
    assert sum(p.numel() for p in model.parameters()) == 2870823808
    assert sum(p.numel() for n, p in model.named_parameters() if compressed_parameter(n)) == 114475264


def test_export_preserves_fp32_bits_metadata_and_reload(tmp_path):
    model = model_fixture()
    owner = pipeline_methods()
    owner.compressed_modulation = True
    owner.transformer_config = dict(hidden_size=16, modulation_rank=4)
    owner.model_config = {'dtype': torch.bfloat16}
    assert owner.training_communication_dtype(torch.bfloat16) == torch.float32
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    # Execute the real generic Saver conversion before the model export.
    tree = ast.parse((ROOT / 'utils/saver.py').read_text(encoding='utf-8'))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'convert_state_dict_dtype')
    ns = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), '<saver>', 'exec'), ns)
    ns['convert_state_dict_dtype'](state, torch.bfloat16, owner.model_save_dtype)
    owner.save_model(tmp_path, state)
    path = tmp_path / 'diffusion_pytorch_model.safetensors'
    assert read_transformer_config(path) == owner.transformer_config
    exported = load_file(path)
    for name, original in model.state_dict().items():
        if compressed_parameter(name):
            assert exported[name].dtype == torch.float32
            torch.testing.assert_close(exported[name], original, rtol=0, atol=0)
        else:
            assert exported[name].dtype == torch.bfloat16
    reloaded = model_fixture()
    reloaded.load_state_dict(exported, strict=True, assign=True)
    assert reloaded.modulation_down.weight.dtype == torch.float32
    assert (tmp_path / 'config.json').is_file()


@pytest.mark.parametrize('architecture,rank', [(None, 4), ('unknown-v2', 4), (ARCHITECTURE, 5)])
def test_rejects_unknown_format_or_rank_mismatch(tmp_path, architecture, rank):
    metadata = {'model_config': json.dumps(dict(hidden_size=16, modulation_rank=rank))}
    if architecture is not None:
        metadata['architecture'] = architecture
    path = tmp_path / 'model.safetensors'
    save_file({'modulation_down.weight': torch.zeros(4, 16)}, path, metadata=metadata)
    with pytest.raises(ValueError):
        read_transformer_config(path)


def test_dense_config_and_invalid_precision(tmp_path):
    path = tmp_path / 'model.safetensors'
    save_file({'dense.weight': torch.zeros(2, 2)}, path)
    (tmp_path / 'config.json').write_text('{"hidden_size": 16}')
    assert read_transformer_config(path) == {'hidden_size': 16}
    with pytest.raises(ValueError, match='compressed_adaln_dtype'):
        resolve_modulation_dtype({'compressed_adaln_dtype': 'float16'})


@pytest.mark.skipif(not torch.cuda.is_available() or os.getenv('RUN_MAGEFLOW_DEEPSPEED_TESTS') != '1',
                    reason='opt-in CUDA/DeepSpeed test; launch with torchrun')
def test_deepspeed_mixed_dtype_steps_and_resume(tmp_path):
    import deepspeed
    from functools import partial
    from utils.pipeline import ManualPipelineModule

    deepspeed.init_distributed(dist_backend='nccl', auto_mpi_discovery=False)
    model = model_fixture().bfloat16()
    layers = layers_for(model)
    # CPU reference tests strip the production CUDA decorators; restore them here.
    for layer in layers:
        layer.forward = torch.autocast('cuda', dtype=torch.bfloat16)(layer.forward)
    pipe = ManualPipelineModule(
        layers=layers, num_stages=1, loss_fn=lambda pred, target: F.mse_loss(pred.float(), target.float()),
        activation_checkpoint_interval=1, checkpointable_layers=['TransformerLayer'],
        activation_checkpoint_func=partial(checkpoint, use_reentrant=False),
    )
    engine, _, _, _ = deepspeed.initialize(
        model=pipe, optimizer=torch.optim.AdamW(pipe.parameters(), lr=1e-4),
        config=dict(train_micro_batch_size_per_gpu=2, gradient_accumulation_steps=2,
                    gradient_clipping=1.0, steps_per_print=100),
    )
    engine._support_torch_style_backward = True
    engine.communication_data_type = torch.float32
    observed = []
    hook = model.modulation_down.weight.register_hook(lambda g: observed.append((g.dtype, bool(g.isfinite().all()))))
    original = model.modulation_down.weight.detach().clone()

    def data():
        while True:
            yield inputs(torch.bfloat16), torch.zeros(2, 4, 4)

    iterator = iter(data())
    for _ in range(2):
        loss = engine.train_batch(data_iter=iterator)
        assert torch.isfinite(loss)
    hook.remove()
    assert observed and all(dtype == torch.float32 and finite for dtype, finite in observed)
    assert not torch.equal(original, model.modulation_down.weight)
    saved = model.modulation_down.weight.detach().clone()
    engine.save_checkpoint(str(tmp_path), tag='mixed')
    with torch.no_grad():
        model.modulation_down.weight.add_(1)
    loaded, _ = engine.load_checkpoint(str(tmp_path), tag='mixed')
    assert loaded is not None
    torch.testing.assert_close(model.modulation_down.weight, saved, rtol=0, atol=0)
    assert model.modulation_down.weight.dtype == torch.float32
