"""CPU numerical regressions; optional real CUDA kernels are tested separately."""

import ast
import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from einops import rearrange

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.mageflow_attention import (
    _packed_attention, packed_attention, packed_attention_metadata, validate_attention_backend,
)
from utils.mageflow_text import encode_text_hidden


def reference_varlen(q, k, v, cu_q, cu_k, max_q, max_k, **kwargs):
    outputs = []
    for start, end in zip(cu_q.tolist()[:-1], cu_q.tolist()[1:]):
        outputs.append(F.scaled_dot_product_attention(
            q[start:end].transpose(0, 1), k[start:end].transpose(0, 1),
            v[start:end].transpose(0, 1)).transpose(0, 1))
    return torch.cat(outputs)


def reference_packed(q, k, v, indices, cu, **kwargs):
    return _packed_attention(q, k, v, indices, cu, reference_varlen)


@pytest.mark.parametrize('keep', [
    [[1, 1, 1, 1, 1]],
    [[1, 0, 0, 1, 1], [1, 1, 1, 1, 1], [0, 0, 0, 1, 1]],
    [[0, 1, 0, 1, 1], [0, 0, 1, 1, 1]],
])
def test_packed_valid_tokens_and_gradients_match_sdpa(keep):
    torch.manual_seed(7)
    mask = torch.tensor(keep, dtype=torch.bool)[:, None, None, :]
    shape = (len(keep), 2, len(keep[0]), 8)
    a = [torch.randn(shape, dtype=torch.float64, requires_grad=True) for _ in range(3)]
    b = [x.detach().clone().requires_grad_() for x in a]
    expected = F.scaled_dot_product_attention(*a, attn_mask=mask)
    actual = reference_packed(*b, *packed_attention_metadata(mask))
    valid_queries = mask.transpose(-1, -2)
    torch.testing.assert_close(actual * valid_queries, expected * valid_queries)
    weight = torch.randn_like(actual) * valid_queries
    (actual * weight).sum().backward()
    (expected * weight).sum().backward()
    for x, y in zip(a, b):
        torch.testing.assert_close(x.grad, y.grad)


def load_training_components():
    # Import the real training functions without requiring DeepSpeed, pretrained
    # files, or the vendored inference stack. Same approach as test_lr_logging.
    tree = ast.parse((ROOT / 'models/mage_flow.py').read_text(encoding='utf-8'))
    names = {'_apply_rope_batched', '_modulate', '_double_stream_block_forward',
             'InitialLayer', 'TransformerLayer', 'FinalLayer'}
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    for node in nodes:
        if isinstance(node, ast.ClassDef):
            for method in node.body:
                if isinstance(method, ast.FunctionDef):
                    method.decorator_list = []  # disable CUDA autocast for CPU reference
    ns = dict(torch=torch, nn=nn, F=F, rearrange=rearrange,
              packed_attention=reference_packed, packed_attention_metadata=packed_attention_metadata,
              encode_text_hidden=encode_text_hidden, make_contiguous=lambda *xs: tuple(x.contiguous() for x in xs))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<mageflow_training>', 'exec'), ns)
    return ns


def tiny_block():
    block = nn.Module()
    for name in ('img_mod', 'txt_mod'):
        setattr(block, name, nn.Linear(16, 96))
    for name in ('img_norm1', 'txt_norm1', 'img_norm2', 'txt_norm2'):
        setattr(block, name, nn.LayerNorm(16, elementwise_affine=False))
    for name in ('img_mlp', 'txt_mlp'):
        setattr(block, name, nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 16)))
    block.attn = nn.Module()
    for name in ('to_q', 'to_k', 'to_v', 'add_q_proj', 'add_k_proj', 'add_v_proj', 'to_add_out'):
        setattr(block.attn, name, nn.Linear(16, 16))
    for name in ('norm_q', 'norm_k', 'norm_added_q', 'norm_added_k'):
        setattr(block.attn, name, nn.LayerNorm(8))
    block.attn.to_out = nn.ModuleList([nn.Linear(16, 16), nn.Identity()])
    return block.double()


@pytest.mark.parametrize('reentrant', [None, False, True])
def test_two_blocks_image_loss_and_weight_gradients(reentrant):
    torch.manual_seed(12)
    forward = load_training_components()['_double_stream_block_forward']
    blocks = nn.ModuleList([tiny_block(), tiny_block()])
    packed_blocks = copy.deepcopy(blocks)
    img, txt, temb = [torch.randn(*shape, dtype=torch.float64, requires_grad=True)
                      for shape in ((3, 4, 16), (3, 5, 16), (3, 16))]
    mask = torch.tensor([[1, 0, 0, 0, 0], [1, 1, 1, 1, 1], [1, 1, 0, 0, 0]], dtype=torch.bool)
    mask = torch.cat([mask, torch.ones(3, 4, dtype=torch.bool)], dim=1)[:, None, None, :]
    freqs = torch.polar(torch.ones(4, 4), torch.randn(4, 4))
    metadata = packed_attention_metadata(mask)

    def run(stack, backend):
        x, text = img, txt
        for block in stack:
            def fn(x, text, emb, block=block):
                return forward(block, x, text, emb, freqs, mask, 2, backend, metadata)
            if reentrant is None:
                text, x = fn(x, text, temb)
            else:
                text, x = checkpoint(fn, x, text, temb, use_reentrant=reentrant)
        return x

    expected = run(blocks, 'sdpa')
    actual = run(packed_blocks, 'reference')
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    expected.square().mean().backward()
    actual.square().mean().backward()
    for (name, a), (_, b) in zip(blocks.named_parameters(), packed_blocks.named_parameters()):
        if a.grad is None:
            assert b.grad is None, name
        else:
            torch.testing.assert_close(a.grad, b.grad, atol=1e-6, rtol=1e-6, msg=name)


def test_qwen_hidden_states_match_original_and_logits_are_reduced():
    transformers = pytest.importorskip('transformers')
    config = transformers.Qwen3VLConfig(
        text_config=dict(vocab_size=64, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         head_dim=8, rope_scaling={'rope_type': 'default', 'mrope_section': [1, 1, 2]}),
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=4,
                           out_hidden_size=32, deepstack_visual_indexes=[]),
        image_token_id=60, video_token_id=61, vision_start_token_id=62, vision_end_token_id=63,
    )
    torch.manual_seed(9)
    encoder = transformers.Qwen3VLForConditionalGeneration(config).eval().requires_grad_(False)
    ids = torch.tensor([[3, 4, 5, 6], [3, 5, 0, 0]])
    mask = ids.ne(0).long()
    calls = []
    hook = encoder.lm_head.register_forward_pre_hook(lambda module, args: calls.append(args[0].shape[1]))
    with torch.no_grad():
        expected = encoder(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states[-1]
    assert calls == [4]
    actual = encode_text_hidden(encoder, ids, mask)
    hook.remove()
    assert calls == [4, 1]
    assert not actual.requires_grad
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_spatial_input_matches_legacy_and_encoder_stays_frozen():
    components = load_training_components()

    class Rope(nn.Module):
        def forward(self, shapes, device):
            _, h, w = shapes[0]
            return torch.ones(h * w, 4, device=device, dtype=torch.complex64)

    class Time(nn.Module):
        def forward(self, t, img):
            return t[:, None].expand(-1, 16)

    model = SimpleNamespace(img_in=nn.Linear(4, 16), txt_norm=nn.LayerNorm(8),
                            txt_in=nn.Linear(8, 16), time_text_embed=Time(), pos_embed=Rope())
    layer = components['InitialLayer'](model)
    img = torch.randn(2, 4, 3, 5)
    rest = (torch.randn(2, 6, 8), torch.ones(2, 6, dtype=torch.bool),
            torch.rand(2), torch.tensor([[3, 5], [3, 5]]))
    a = layer((img, *rest))
    b = layer((rearrange(img, 'b c h w -> b (h w) c'), *rest))
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    layer.text_encoder = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.5)).requires_grad_(False)
    layer.train()
    assert layer.training and not layer.text_encoder.training


def test_backend_validation_preserves_default_and_rejects_unsupported_modes():
    validate_attention_backend('sdpa', 8)
    with pytest.raises(ValueError, match='attention_backend'):
        validate_attention_backend('typo')
    with pytest.raises(ValueError, match='pipeline_stages=1'):
        validate_attention_backend('flash_attn_3', 2)
    with pytest.raises(ValueError, match='CUDA'):
        packed_attention(*(torch.randn(1, 2, 3, 8) for _ in range(3)),
                         torch.arange(3), torch.tensor([0, 3], dtype=torch.int32), 'flash_attn_2')


@pytest.mark.parametrize('backend', ['flash_attn_2', 'flash_attn_3'])
@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA and installed FlashAttention')
def test_real_flash_kernel_forward_and_backward(backend):
    validate_attention_backend(backend)
    torch.manual_seed(19)
    mask = torch.ones(3, 1, 1, 129, dtype=torch.bool, device='cuda')
    mask[0, :, :, 3:35] = False
    mask[1, :, :, :35] = False
    a = [torch.randn(3, 4, 129, 128, device='cuda', dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
    b = [x.detach().clone().requires_grad_() for x in a]
    expected = F.scaled_dot_product_attention(*a, attn_mask=mask)
    actual = packed_attention(*b, *packed_attention_metadata(mask), backend)
    valid = mask.transpose(-1, -2)
    torch.testing.assert_close(actual * valid, expected * valid, rtol=2e-2, atol=2e-2)
    grad = torch.randn_like(actual) * valid
    expected.backward(grad)
    actual.backward(grad)
    for x, y in zip(a, b):
        torch.testing.assert_close(x.grad, y.grad, rtol=5e-2, atol=5e-2)
