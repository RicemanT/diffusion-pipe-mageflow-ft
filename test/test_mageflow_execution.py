"""Checkpoint-selection, compile/autograd, and checkpoint-key regressions."""

import ast
import copy
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.mageflow_execution import validate_execution_config, compile_block_forward
from test.test_mageflow_training import load_training_components, tiny_block


@pytest.fixture(scope='module', autouse=True)
def small_tensor_thread_count():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def config(**model):
    return {'activation_checkpointing': True, 'model': model}


@pytest.mark.parametrize('indices', [[], [0], [0, 2, 5], list(range(12))])
def test_checkpoint_selection_valid(indices):
    validate_execution_config(config(checkpoint_blocks=indices), 12)


@pytest.mark.parametrize('indices', [None, True, 'all', [True], [-1], [0.5], [1, 1], [12]])
def test_checkpoint_selection_rejects_invalid_indices(indices):
    with pytest.raises(ValueError, match='checkpoint_blocks'):
        validate_execution_config(config(checkpoint_blocks=indices), 12)


@pytest.mark.parametrize('overrides,model', [
    ({'activation_checkpointing': False}, {'checkpoint_blocks': []}),
    ({'activation_checkpointing': 'unsloth'}, {'checkpoint_blocks': [0]}),
    ({'blocks_to_swap': 2}, {'checkpoint_blocks': [0]}),
    ({'compile': True}, {'compile_blocks': True}),
    ({'blocks_to_swap': 2}, {'compile_blocks': True}),
    ({'activation_checkpointing': 'unsloth'}, {'compile_blocks': True}),
    ({'optimizer': {'gradient_release': True}}, {'compile_blocks': True}),
    ({}, {'compile_blocks': 'true'}),
    ({}, {'compile_blocks': True, 'attention_backend': 'flash_attn_3'}),
    ({}, {'compile_blocks': True, 'compile_mode': 'reduce-overhead'}),
    ({}, {'compile_blocks': True, 'compile_dynamic': 'true'}),
    ({}, {'compile_mode': 'default'}),
])
def test_incompatible_options_fail_early(overrides, model):
    cfg = config(**model)
    cfg.update(overrides)
    with pytest.raises(ValueError):
        validate_execution_config(cfg, 12)


def test_existing_config_defaults_unchanged():
    validate_execution_config(config(), 12)
    validate_execution_config({'model': {}, 'activation_checkpointing': 'unsloth', 'compile': True})
    validate_execution_config(config(compile_blocks=True, checkpoint_blocks=[0, 2]), 12)


def test_compile_options_are_explicit(monkeypatch):
    options = []
    monkeypatch.setattr(torch, 'compile', lambda fn, **kwargs: options.append(kwargs) or fn)
    fn = lambda: None
    assert compile_block_forward(fn, {}) is fn
    assert options == [{'dynamic': True, 'mode': 'default', 'fullgraph': True}]
    compile_block_forward(fn, {'compile_dynamic': False, 'compile_mode': 'max-autotune-no-cudagraphs'})
    assert options[-1] == {'dynamic': False, 'mode': 'max-autotune-no-cudagraphs', 'fullgraph': True}


def pipeline_class():
    # Exercise the actual override without importing DeepSpeed's CUDA stack.
    tree = ast.parse((ROOT / 'utils/pipeline.py').read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ManualPipelineModule')

    class Base:
        def _is_checkpointable(self, funcs):
            return all(type(f).__name__ in self.checkpointable_layers for f in funcs)

    ns = {'PipelineModule': Base}
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<pipeline>', 'exec'), ns)
    cls = ns['ManualPipelineModule']
    module = cls.__new__(cls)
    module.checkpointable_layers = ['TransformerLayer']
    return module


class Offloader:
    def wait_for_block(self, idx):
        pass

    def submit_move_blocks_forward(self, idx):
        pass


def inputs(text_len=5, image_len=4, device='cpu', dtype=torch.float64):
    mask = torch.ones(2, 1, 1, text_len + image_len, device=device, dtype=torch.bool)
    mask[0, :, :, 1:text_len] = False
    return (
        torch.randn(2, image_len, 16, device=device, dtype=dtype, requires_grad=True),
        torch.randn(2, text_len, 16, device=device, dtype=dtype, requires_grad=True),
        mask,
        torch.randn(2, 16, device=device, dtype=dtype, requires_grad=True),
        torch.polar(torch.ones(image_len, 4, device=device), torch.randn(image_len, 4, device=device)),
    )


def test_pipeline_selection_keeps_layer_identity_and_eligibility():
    layer_cls = load_training_components()['TransformerLayer']
    layer = layer_cls(tiny_block(), 0, 2, Offloader())
    pipeline = pipeline_class()
    assert pipeline._is_checkpointable([layer])
    before = list(layer.state_dict())
    layer.checkpoint_enabled = False
    assert not pipeline._is_checkpointable([layer])
    assert not pipeline._is_checkpointable([nn.Linear(2, 2)])
    assert list(layer.state_dict()) == before


def test_to_layers_wires_selection_and_excludes_terminal_block(monkeypatch):
    ns = load_training_components()
    tree = ast.parse((ROOT / 'models/mage_flow.py').read_text(encoding='utf-8'))
    pipeline = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MageFlowPipeline')
    method = next(n for n in pipeline.body if isinstance(n, ast.FunctionDef) and n.name == 'to_layers')
    compiled = lambda *args: None
    ns.update(validate_execution_config=validate_execution_config,
              compile_block_forward=lambda *args: compiled,
              InitialLayer=lambda *args, **kwargs: nn.Identity(),
              FinalLayer=lambda *args: nn.Identity(), PROMPT_TEMPLATE_ENCODE_START_IDX=34)
    exec(compile(ast.Module(body=[method], type_ignores=[]), '<to_layers>', 'exec'), ns)
    cfg = config(compile_blocks=True, checkpoint_blocks=[0, 2])
    model = SimpleNamespace(config=cfg, model_config=cfg['model'], cache_text_embeddings=True,
                            transformer=SimpleNamespace(transformer_blocks=[tiny_block() for _ in range(3)],
                                                        num_attention_heads=2),
                            attention_backend='sdpa', attention_deterministic=False, offloader=Offloader())
    layers = ns['to_layers'](model)[1:-1]
    assert [layer.checkpoint_enabled for layer in layers] == [True, False, True]
    assert [layer.compiled_forward is compiled for layer in layers] == [True, True, False]
    assert all(layer.block is block for layer, block in zip(layers, model.transformer.transformer_blocks))


@pytest.mark.parametrize('reentrant', [False, True])
def test_only_selected_blocks_recompute_with_matching_accumulated_gradients(reentrant):
    torch.manual_seed(93)
    layer_cls = load_training_components()['TransformerLayer']
    layers = nn.ModuleList([layer_cls(tiny_block(), i, 2, Offloader(), checkpoint_enabled=(i == 0))
                            for i in range(2)])
    reference = copy.deepcopy(layers)
    counts = [0, 0]
    hooks = []
    for i, layer in enumerate(layers):
        def count(module, args, i=i):
            counts[i] += 1
        hooks.append(layer.block.img_mod.register_forward_pre_hook(count))
    pipeline = pipeline_class()
    for _ in range(4):  # user's accumulation count
        batch = inputs()
        x, expected = batch, batch
        for layer, ref in zip(layers, reference):
            if pipeline._is_checkpointable([layer]):
                x = checkpoint(lambda *xs, layer=layer: layer(xs), *x, use_reentrant=reentrant)
            else:
                x = layer(x)
            expected = ref(expected)
        torch.testing.assert_close(x[0], expected[0])
        (x[0].square().mean() / 4).backward()
        (expected[0].square().mean() / 4).backward()
    for hook in hooks:
        hook.remove()
    assert counts == [8, 4]
    for a, b in zip(layers.parameters(), reference.parameters()):
        if a.grad is None:
            assert b.grad is None
        else:
            torch.testing.assert_close(a.grad, b.grad)


@pytest.mark.parametrize('reentrant', [False, True])
def test_compiled_blocks_dynamic_shapes_checkpoint_backward_and_resume(reentrant):
    # aot_eager exercises Dynamo full-graph capture and AOTAutograd on CPU.
    # CUDA Inductor gets a separate test below; CPU capture is not a speed claim.
    torch._dynamo.reset()
    torch.manual_seed(23)
    ns = load_training_components()
    forward = ns['_double_stream_block_forward']
    compiled = torch.compile(forward, backend='aot_eager', dynamic=True, fullgraph=True)
    cls = ns['TransformerLayer']
    eager = cls(tiny_block(), 0, 2, Offloader())
    candidate = cls(copy.deepcopy(eager.block), 0, 2, Offloader(), compiled_forward=compiled)
    # Real training compiles intermediate blocks, whose text output feeds the
    # next block. The terminal block stays eager to preserve None gradients.
    last = cls(tiny_block(), 1, 2, Offloader())
    last_candidate = copy.deepcopy(last)
    original_keys = list(eager.state_dict())
    for text_len, image_len in [(5, 4), (7, 6)]:
        eager.zero_grad(set_to_none=True)
        candidate.zero_grad(set_to_none=True)
        last.zero_grad(set_to_none=True)
        last_candidate.zero_grad(set_to_none=True)
        batch = inputs(text_len, image_len)
        expected = last(eager(batch))[0]
        actual = last_candidate(checkpoint(lambda *xs: candidate(xs), *batch, use_reentrant=reentrant))[0]
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
        expected.square().mean().backward()
        actual.square().mean().backward()
        for a, b in zip(list(eager.parameters()) + list(last.parameters()),
                        list(candidate.parameters()) + list(last_candidate.parameters())):
            if a.grad is None:
                assert b.grad is None
            else:
                torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=2e-6)
    assert list(candidate.state_dict()) == original_keys
    eager.load_state_dict(candidate.state_dict(), strict=True)
    candidate.load_state_dict(eager.state_dict(), strict=True)
    # Compilation is training-only; eval/sampling uses the established eager path.
    candidate.eval()
    candidate.compiled_forward = lambda *args: pytest.fail('eval invoked compiled training graph')
    with torch.no_grad():
        candidate(inputs())
    torch._dynamo.reset()


def vendored_block(monkeypatch, dim=16, heads=2):
    pytest.importorskip('diffusers')
    # Load the actual module under a local test package, avoiding Mage's
    # inference-only package __init__ and its unrelated serving dependencies.
    name = '_mageflow_execution_test_modules'
    package = ModuleType(name)
    package.__path__ = [str(ROOT / 'Mage/mage_flow/models/modules')]
    monkeypatch.setitem(sys.modules, name, package)
    spec = importlib.util.spec_from_file_location(
        name + '.mage_layers', ROOT / 'Mage/mage_flow/models/modules/mage_layers.py')
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module.MageFlowTransformerBlock(dim, heads, dim // heads)


def test_actual_vendored_block_fullgraph_and_gradients(monkeypatch):
    torch._dynamo.reset()
    torch.manual_seed(37)
    ns = load_training_components()
    cls = ns['TransformerLayer']
    original = vendored_block(monkeypatch).double()
    eager = cls(original, 0, 2, Offloader())
    compiled = torch.compile(ns['_double_stream_block_forward'], backend='aot_eager', dynamic=True, fullgraph=True)
    candidate = cls(copy.deepcopy(original), 0, 2, Offloader(), compiled_forward=compiled)
    eager_second = cls(copy.deepcopy(original), 1, 2, Offloader())
    candidate_second = cls(copy.deepcopy(original), 1, 2, Offloader(), compiled_forward=compiled)
    # Separate weights through one shared compiled callable must stay separate.
    with torch.no_grad():
        next(eager_second.parameters()).mul_(0.7)
        candidate_second.load_state_dict(eager_second.state_dict())
    # Use both streams as downstream inputs, as in an intermediate block.
    batch = inputs()
    a = eager_second(eager(batch))
    first = checkpoint(lambda *xs: candidate(xs), *batch, use_reentrant=False)
    b = checkpoint(lambda *xs: candidate_second(xs), *first, use_reentrant=False)
    for x, y in zip(a[:2], b[:2]):
        torch.testing.assert_close(x, y, atol=2e-6, rtol=2e-6)
    (a[0].square().mean() + a[1].square().mean()).backward()
    (b[0].square().mean() + b[1].square().mean()).backward()
    for x, y in zip(list(eager.parameters()) + list(eager_second.parameters()),
                    list(candidate.parameters()) + list(candidate_second.parameters())):
        torch.testing.assert_close(x.grad, y.grad, atol=2e-6, rtol=2e-6)
    torch._dynamo.reset()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA Inductor')
def test_cuda_inductor_sdpa_checkpoint_gradients():
    torch.manual_seed(31)
    ns = load_training_components()
    cls = ns['TransformerLayer']
    eager = cls(tiny_block().to(device='cuda', dtype=torch.bfloat16), 0, 2, Offloader())
    candidate = cls(copy.deepcopy(eager.block), 0, 2, Offloader(),
                    compiled_forward=compile_block_forward(ns['_double_stream_block_forward'], {}))
    last = cls(tiny_block().to(device='cuda', dtype=torch.bfloat16), 1, 2, Offloader())
    last_candidate = copy.deepcopy(last)
    batch = inputs(device='cuda', dtype=torch.bfloat16)
    expected = last(eager(batch))[0]
    actual = last_candidate(checkpoint(lambda *xs: candidate(xs), *batch, use_reentrant=False))[0]
    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)
    expected.float().square().mean().backward()
    actual.float().square().mean().backward()
    for a, b in zip(eager.parameters(), candidate.parameters()):
        if a.grad is None:
            assert b.grad is None
        else:
            torch.testing.assert_close(a.grad, b.grad, atol=5e-2, rtol=5e-2)
