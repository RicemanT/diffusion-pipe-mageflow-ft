# MageFlow performance and safety audit

Subsequent optimizer correctness work is documented in
[AdamW8bitKahan correctness and precision](adamw8bitkahan.md), including the
weight-decay fix and optional stochastic rounding. The unchanged-optimizer
statements below describe the original performance audit only.

## Scope and compatibility

This pass targets the supplied 8-H100 full-finetune setup: `pipeline_stages=1`,
microbatch 16, accumulation 4, BF16 transformer, AdamW8bitKahan, StageLR,
activation checkpointing, and live NF4 Qwen conditioning with up to 1,280 tokens.
With eight ranks the effective batch is 512 images per optimizer step.

No existing TOML files, optimizer implementation, scheduler, caption
augmentation, attribution rules, sampling cadence, trainable parameters, or
state-dict names were changed. Default attention remains SDPA. The original
token-layout input remains accepted by the initial layer for external callers.

The pasted `total_iters = #have to calculate yourself` is not valid TOML until
filled in. Its value needs the processed bucket counts and intended continuation
schedule; no value has been guessed. Loading `transformer_path` starts from model
weights; it does not itself restore Adam moments, Kahan compensation, dataloader
position, or StageLR progress. Full continuation uses the trainer's existing
`--resume_from_checkpoint` mechanism.

## Implemented changes

| Change | Why it helps | Validation / limits |
| --- | --- | --- |
| Qwen `logits_to_keep=1`, `use_cache=False` | Reduces unused vocabulary projection from every token to one position; avoids generating an unused KV cache on every caption batch | Tiny real Qwen3-VL test verifies bitwise identical original `hidden_states[-1]`, including padded input. NF4 CUDA must still be tested on the deployment stack. |
| Keep the frozen Qwen in eval mode when the pipeline enters train mode | Stops recursive `.train()` from enabling dropout in the frozen conditioner | Unit test checks recursive mode handling. Stock Qwen attention dropout is zero. |
| Derive image height/width from spatial tensor shapes | Removes two GPU `.item()` reads per initial-layer call on the internal training/sampling paths | Spatial and legacy token inputs produce identical initial-layer outputs. |
| Opt-in packed FlashAttention 2/3 | Removes padded queries and keys from joint attention and explicitly selects a fused kernel | Valid outputs and gradients checked against SDPA, including two blocks and both checkpoint modes. Real CUDA tests are supplied but not run locally. |
| Build packed indices and cumulative lengths once per microbatch | Avoids repeated metadata construction during each block and backward recomputation | Metadata remains integer tensors outside autograd; packing still has a one-time CUDA synchronization. |
| Integer epoch MAX reduction | Replaces a per-step Python-object all-gather with one scalar all-reduce | Two-process Gloo test; NCCL remains to be exercised on H100. |
| Rank-zero checkpoint signal decision, broadcast to all ranks | Prevents ranks from diverging when signal files become visible at different times | Tests cover nonzero ranks with no filesystem reads, save/quit, failed saves, and request priority. |
| Convert adapter export dtype once after collecting tensors | Removes repeatedly walking the growing state dict | Does not affect full-finetune exports. |

Qwen compatibility deserves particular care. In Transformers 4.57.6, the outer
generation wrapper's `hidden_states[-1]` is the final decoder output before the
last RMSNorm; the backbone's output-capture wrapper substitutes a normalized
state. Simply calling `.model(...).last_hidden_state`, or even the backbone's
`hidden_states[-1]`, changes conditioning. The implementation keeps the original
outer wrapper and hidden-state selection. It still collects hidden states and
computes one position of unused logits to preserve this behavior and avoid
zero-length GEMMs in quantized layers. Frozen weight storage is unchanged.

## SDPA: selective block checkpointing and compilation

These are the current development focus. Neither requires FlashAttention.
Existing configurations continue to use all-block checkpointing when
`activation_checkpointing=true`, with block compilation disabled.

Start by testing compilation with **all existing checkpointing retained**. Add
these keys to the existing `[model]` table (do not create a second table):

```toml
compile_blocks = true
compile_dynamic = true
compile_mode = 'default'
```

Keep top-level `compile` false or commented out. DeepSpeed's existing top-level
compile path compiles every pipeline layer, including the initial layer with
live Qwen. The new option compiles only the transformer block computation.
Qwen NF4, tokenization, offloader calls, initial/final projections, and sampling
wrappers remain outside the compiled graph. SDPA remains selected.

Next, if measured peak memory leaves enough room, add a checkpoint selection
under the same `[model]` table. For the 12-block model, a conservative first
experiment is retaining checkpointing on 10 blocks:

```toml
# Checkpoint blocks 0..9; blocks 10 and 11 retain activations.
checkpoint_blocks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```

The list identifies blocks to **recompute during backward**, using zero-based
global transformer indices. Omit the key for the original all-block behavior;
`[]` means no transformer block is checkpointed. Unlisted blocks retain their
intermediate activations: fewer checkpointed blocks saves recomputation but
increases VRAM. At microbatch 16 and 1,280 text tokens, even removing two blocks
from checkpointing may exceed available memory. No automatic choice is made.
Keep top-level `activation_checkpointing=true` when using the list. The list
can be used independently of compilation and with native reentrant or
non-reentrant checkpointing. This is selection of **whole blocks**, not PyTorch's
operation-level selective activation checkpointing policy.

Compilation uses `fullgraph=True`: an unsupported graph fails visibly instead
of silently accumulating graph breaks. `compile_dynamic=true` requests dynamic
shapes for text length/resolution variation; it cannot guarantee that all shapes
share a graph. Diagnose recompiles with `TORCH_LOGS=recompiles`. The optional
`compile_mode='max-autotune-no-cudagraphs'` spends more startup time tuning
kernels; measure whether it pays off for your run. Modes that enable CUDA graphs
are excluded here because accumulation can keep several forwards live.

The compiled path expresses image RoPE with real arithmetic; frequency-table
construction and complex-to-real view conversion happen outside the graph.
Parameter/module registration is unchanged: no `_orig_mod` prefixes are added
to checkpoints or exports. Compile caches are runtime artifacts and rebuilt
on resume. The existing optimizer, schedule, accumulation, and SDPA settings are
not modified.

**The final transformer block remains eager.** AOTAutograd can materialize zero
gradients for an unused text output instead of the original `None` gradients.
At the terminal block those differences can activate optimizer momentum/decay
updates on otherwise unused text branches. Keeping that block eager preserves
the original behavior while compiling the preceding 11 blocks. Both checkpoint
modes and the eager terminal block are covered by gradient tests. Sampling
continues through the existing eager wrappers and avoids extra compilation.

Invalid/duplicate/out-of-range indices fail configuration validation. Selection
with block swapping or Unsloth checkpointing is rejected. The new compilation
path rejects block swapping, Unsloth checkpointing, gradient release, and
non-SDPA attention. These combinations are outside the tested scope; your
supplied full-finetune configuration does not enable them.

For a short H100 comparison, keep model, dataset, captions, batch size,
accumulation, optimizer, and seeds fixed, and use separate output directories:

| Run | `compile_blocks` | `checkpoint_blocks` |
| --- | --- | --- |
| Existing baseline | false / omitted | omitted |
| Compilation only | true | omitted |
| Less recomputation only | false / omitted | `[0,1,2,3,4,5,6,7,8,9]` |
| Combined | true | `[0,1,2,3,4,5,6,7,8,9]` |

Record startup compilation time separately from warmed steps, and cover the
largest actual training bucket before deciding on a memory budget. Compare
images/second, peak allocated/reserved VRAM, finite losses/gradients, and a
checkpoint resume round trip. There is no measured H100 speedup yet.

```bash
python -m pytest test/test_mageflow_execution.py -q
```

CPU tests exercise full-graph Dynamo/AOTAutograd, changing sequence lengths,
native checkpoint recomputation, accumulated gradients, state-dict compatibility,
and configuration errors. Tests also use the actual vendored MageFlow block
with Diffusers 0.35.2, including different block weights sharing one compiled
callable. The expanded regression run completed with **161 passed, 3 skipped**;
the skips are CUDA Inductor and the two previously added FlashAttention tests.
Python syntax checks, dependency consistency, and `git diff --check` passed.
A separate CUDA test exercises Inductor with SDPA and
BF16; it is skipped on the local CPU-only environment. Actual DeepSpeed/NCCL
eight-GPU throughput and resume still need the deployment environment.

## Previously added opt-in attention (not required for the SDPA work)

Leave the supplied config as it is for the default improvements. To experiment
with the Hopper kernel, add under the existing `[model]` table:

```toml
attention_backend = 'flash_attn_3'
# Optional: deterministic backward, with extra runtime/memory cost.
# attention_deterministic = true
```

Supported values are `sdpa` (default), `flash_attn_2`, and `flash_attn_3`.
The selected external kernel must be installed; import failures are explicit,
with no silent SDPA fallback. FA3 uses the current official
`flash_attn_3.flash_attn_interface` package path. Older installations exposing
only `flash_attn_interface` need the matching current package installation.
FA3 requires Hopper-class supported hardware and CUDA per its upstream README;
the fetched README recommends CUDA 12.8 and lists CUDA 12.3 as the minimum.

Packed attention currently requires `pipeline_stages=1`, which includes your
eight-way data-parallel setup. Multi-stage packed attention is rejected because
packed tensor sizes can differ between microbatches while the current trainer
resets DeepSpeed receive shapes only once per step. Existing multi-stage SDPA
behavior is unchanged. Block swapping and compile combinations need separate
CUDA validation before deployment.

The packing change is confined to attention: projections, MLPs, modulation, and
residuals still operate on the original batched layout. Padded query outputs are
zeroed; those positions are excluded from later attention keys and from the
image loss. Numerical rounding differs between attention kernels; bitwise
training trajectory identity is not promised. Fully packing MLPs as well is a
possible later optimization that needs more extensive architecture and compiler
work.

## Validation on the training machine

Local checks use CPU PyTorch 2.14.0 and Transformers 4.57.6 on Windows, without
pretrained weights or DeepSpeed. They establish numerical/layout and control-flow
behavior, not H100 performance or full distributed-training compatibility.

Result: **128 passed, 2 skipped** across the new MageFlow/checkpoint tests and
existing LR logging, validation sampling, sampling driver, tracking, and W&B step
alignment tests. The skipped cases require actual CUDA FA2/FA3 kernels. Python
syntax compilation and `git diff --check` also passed; benchmark CLI help and its
CPU rejection path were checked. Test dependencies were installed only in the
ignored repository `.venv`; project requirements were not modified.

Run the regression tests on Linux with your pinned training environment:

```bash
python -m pytest test/test_mageflow_training.py test/test_checkpoint_control.py -q
```

The two CUDA tests explicitly request FA2 and FA3. Install both to run both, or
select the backend you installed with pytest's `-k` expression. When CUDA is
available, a missing requested kernel fails rather than silently skipping it.

Benchmark isolated attention with the actual transformer config so head count
and dimension are not guessed:

```bash
python tools/benchmark_mageflow_attention.py \
  --transformer-config /workspace/Mage-Flow/transformer/config.json \
  --batch-size 16 --height 1216 --width 832 \
  --text-tokens 1280 --min-text-tokens 128 \
  --backends sdpa flash_attn_3
```

This reports forward+backward wall time, allocated/reserved CUDA memory, and
metadata construction time. It includes Q/K/V pack/unpack and uses synthetic
mixed caption lengths. It excludes projections, MLPs, Qwen, recomputation,
optimizer, and inter-GPU communication: do not report its ratio as training
speedup. Change dimensions and caption length ranges to match real buckets.

Before a long run, use separate output directories and the same starting model
for short eight-GPU baseline and candidate jobs. Keep all training settings,
dataset order/seeds, GPU allocation, and dependency versions fixed. Record:

- Warm steady-state seconds per optimizer step and images/second (512 / seconds
  for the supplied batch settings); measure startup/caching and sampling separately.
- Peak VRAM per rank, actual bucket sizes and caption lengths, and time in Qwen,
  DiT forward/backward, gradient reductions, and AdamW8bitKahan.
- Finite losses and gradients, fixed-seed validation samples, and a save/resume
  round trip restoring optimizer, scheduler, and loader state.
- Output of `pip freeze`, CUDA/driver versions, GPU memory capacity, and NVLink
  topology. Pin the tested stack before committing the long run.

No throughput percentage or cost saving has been measured on H100 in this pass.

## Next investigations, without changing the supplied recipe

1. **Profile AdamW8bitKahan and bitsandbytes.** Per-parameter dispatch and library
   synchronization may dominate a fast H100 step. Preserve quantized moments,
   Kahan compensation, weight-decay semantics, and checkpoint state while testing
   any fused/grouped replacement. No optimizer rewrite was made here.
2. **Selective activation checkpointing and block compilation.** Whole-block
   selection and SDPA block compilation are implemented above. Next is H100
   validation of throughput, memory, and recompilation behavior. Operation-level
   checkpoint policies remain a separate potential extension.
3. **Input preparation.** CPU noise generation, batched caption processing, and
   pageable host transfers deserve profiling. Moving RNG to CUDA or changing
   batching can change stochastic trajectories; neither was silently changed.
4. **Checkpoint durability and resume reproducibility.** Signal coordination is
   fixed, but this is not transactional checkpoint publication or exact replay of
   prefetched stochastic captions/noise. The supplied config comments out timed
   training-state checkpoints; model exports alone cannot restore Adam/StageLR.
   Choosing a checkpoint interval requires storage throughput and job-failure
   assumptions, so no cadence was changed.
5. **FlashAttention 4 / FP8 / alternative sharding.** Current official FA4 docs
   mention Hopper support. Test its backward and compiler behavior before adding
   another dependency path. Casting trainable weights to FP8 is not equivalent to
   a supported scaled FP8 training recipe. Eight-way data parallelism already
   suits this configuration; replacing it with pipeline stages would change
   accumulation efficiency and sampling support.

## Sources inspected (2026-09-09)

- [PyTorch SDPA API and backend-selection notes](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html).
  Masked SDPA does not necessarily mean the math kernel: available fused kernels
  depend on shapes, masks, dtype, and PyTorch/CUDA versions. Profile the actual path.
- [Official FlashAttention README](https://github.com/Dao-AILab/flash-attention/blob/main/README.md)
  and [Hopper varlen interface](https://github.com/Dao-AILab/flash-attention/blob/main/hopper/flash_attn_interface.py).
- [Official Qwen3-VL implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py),
  plus installed Transformers 4.57.6 source and a real tiny-model numerical test.
- [DeepSpeed 0.18.4 pipeline engine](https://github.com/deepspeedai/DeepSpeed/blob/v0.18.4/deepspeed/runtime/pipe/engine.py),
  matching this repository's requirements pin.
- Vendored MageFlow attention/RoPE/model implementation under `Mage/mage_flow`,
  the trainer's data pipeline, save paths, StageLR setup, and supplied config.

URLs pointing at `main` are research references, not dependency pins.
