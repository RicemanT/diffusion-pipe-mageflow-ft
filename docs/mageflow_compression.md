# Compressed MageFlow training

This trainer now loads Bluvoll's `mageflow-lowrank-modulation-v1` transformer
format, including the published rank-256 MageTrail checkpoint. The integration
keeps the shared projection and per-block heads trainable, with FP32 parameters,
gradients, and projection arithmetic by default. The primary deployment target
is the existing single-H100 full-finetuning workflow.

CPU numerical validation is available; a real H100/DeepSpeed training run,
throughput measurement, and long-run quality evaluation remain required.
On 2026-09-17, all 399 published tensor keys and shapes were checked against the
constructed architecture using the real checkpoint header, without downloading
the tensor payload. The compression suite passed 17 tests and skipped its opt-in
CUDA/DeepSpeed test. The existing combined MageFlow regression run passed 55
tests with three CUDA-only skips; that run included an earlier subset of the
compression tests. Python syntax, example TOML and patch whitespace checks passed.

## What changed in the model

The checkpoint is a complete transformer, not a LoRA or a quantized dense model.
For each timestep embedding `t`, it computes:

```text
x = SiLU(t)                          # original trunk precision
z = modulation_down(x)               # one shared hidden_size -> rank projection
image_mod[i] = image_head[i](z)        # rank -> 6 * hidden_size
text_mod[i]  = text_head[i](z)         # separate head for each block and stream
```

The head results return to each stream's activation dtype before modulation.
The final output AdaLN still receives the original `t`, not `z`.
For rank 256, hidden size 3072 and 12 blocks, this replaces 1,359,396,864 block
modulation parameters with 114,475,264 parameters. Total transformer parameters
fall from 4,115,745,408 to 2,870,823,808 (about 30.25%). FP32 compressed factors
occupy about 436.69 MiB. Relative to BF16 dense factors, weight storage alone
saves about 2.11 GiB; gradients and optimizer state offer additional savings.
Activations and a resident text encoder are separate costs.

## Why DeepSpeed does not prevent this

The screenshots describe a concern that is valid for some global mixed-precision
configurations, but not a blanket limitation of DeepSpeed. This repository's
`train.py` does not enable DeepSpeed's global `bf16`/`fp16` optimizer modes or
ZeRO. It loads parameter dtypes itself and uses per-layer CUDA autocast.

In the pinned DeepSpeed 0.18.4 engine, `_configure_distributed_model` only calls
`.bfloat16()` when global BF16 is enabled; the fallback parameter check is a
no-op. Its normal gradient reduction splits tensors into dtype buckets. However,
`allreduce_bucket` casts to `communication_data_type`, which this trainer
previously set to the model dtype. Compressed training now selects FP32
communication to avoid rounding FP32 gradients, including during data parallelism.
This does not add FP32 accumulation buffers for BF16 trunk parameters.

FP32 storage alone is insufficient: autocast would still execute ordinary linear
projections in BF16. `ModulationProjection` disables autocast locally and converts
its inputs to its parameter dtype. It also preserves coefficients across module
dtype/device moves without a temporary BF16 round-trip. FP32 matmul can still use
TF32 if the deployment enables it; use full FP32 matmul as the numerical baseline.

The initial pipeline layer owns the shared projection exactly once and passes
`z` through subsequent layers as an activation. This avoids duplicate optimizer
parameters and checkpoint aliases, and preserves gradient accumulation from all
heads. Pipeline tensors carry both `t` and `z`; block checkpointing recomputes
heads without recomputing the shared projection. The tuple layout also preserves
packed-attention metadata.

## Loading, saving, and existing features

The loader reads safetensors `architecture` and `model_config` metadata, validates
the rank against the shared weight shape, constructs the compressed modules on
the meta device, and loads with strict key/shape checking. It never allocates the
discarded dense modulation weights. Unknown compression formats fail explicitly.
Ordinary dense MageFlow continues to use its adjacent `config.json`.

Exports retain `architecture = "mageflow-lowrank-modulation-v1"`, embedded model
configuration, the original tensor names, and FP32 factors. A model-specific
save-dtype hook runs before the generic saver can round weights through BF16.
An adjacent `config.json` is also written. These exports meet the dedicated
ComfyUI loader's metadata, key and factor-dtype requirements; actual ComfyUI
sampling of a newly trained export remains a deployment check.
The standalone `tools/mage_infer.py` utility still uses the dense upstream loader;
use the trainer's validation sampling or the dedicated ComfyUI compressed loader.

`model.compressed_adaln_dtype = 'bfloat16'` is an explicit experimental opt-in.
Its gradients and projection arithmetic use BF16. Export still stores factors
in FP32 for the ComfyUI loader; upcasting does not recover lost precision.
Keep the same dtype setting when resuming a training checkpoint.

Full finetuning trains all remaining parameters. Generic PEFT/LyCORIS target
discovery only selects `nn.Linear`; compressed projections are purpose-built
modules, so adapter training leaves them frozen. Adapters target the attention
and MLP trunk and need their matching compressed base. Adapter execution and
block swapping with compressed models have not been GPU-validated here.

Existing caption preparation, cached/live text conditioning, StageLR, validation
sampling and optimizer selection remain in the same pipeline. Cached embeddings
can be reused when the text encoder, tokenizer and captions are unchanged.
Switching only the transformer does not make cached text stale. Live caption
augmentation still requires the existing live text mode and its encoder memory.

DeepSpeed resume retains the existing mechanism. A compressed run needs its own
optimizer checkpoint: dense-model optimizer states have incompatible shapes.
Loading an exported transformer starts a new run; it does not restore the
optimizer, scheduler or dataloader position. Multi-stage and data-parallel
execution are wired through the existing pipeline but have not been GPU-tested.

## Review of Bluvoll's implementation

Sources reviewed at trainer commit `b2912d7c4c63cf782d8b9453153db6a24b1bf598`,
ComfyUI commit `d442acfdbc535b34af3f0c5846cf8af5a7d1a0d1`, and model revision
`23278f33faadf364bd1d5a279e76b457b03957b6`.

* **Conversion:** `distill_modulation.py` calibrates post-SiLU timestep features,
  subtracts their mean, and uses SVD to find a shared input basis. It does not
  independently factor each block's weight. With basis `B` and mean `m`, the
  down projection is `(B.T, -m @ B)` and each head is
  `(W @ B, b + W @ m)`. This approximates the original projection on the
  calibrated timestep-feature subspace. There is no second SiLU after `z`.
* **Distillation:** optional optimization fits modulation outputs to dense
  teacher outputs, with separate shared/head learning rates and group-normalized
  loss. It is timestep-only distillation, not image training. Held-out timestep
  validation selects the best snapshot. The released FP64/QR rank-256 model is
  calibrated initialization, without gradient-based distillation. The published
  ~1.18% number is relative flow-prediction error on a small probe set, not an
  image-quality percentage or a training stability guarantee.
* **Precision:** his trainer retains FP32 factors through `.to()` operations and
  casts head outputs back to trunk precision. The preservation strategy matters
  because casting to BF16 and back silently loses calibration bits. This
  integration additionally needs explicit autocast exclusions because diffusion
  pipe wraps the layer forward calls in CUDA autocast.
* **SDNQ:** his trainer can quantize supported trunk weights while excluding
  compressed factors. Optimizer-state quantization is a separate option and can
  still apply to factors. His short rank-256 test reports roughly 16.08 GiB
  allocated / 17.01 GiB reserved, using INT8 SDNQ, quantized optimizer state,
  cached text, checkpointing, compilation, batch 1, and no CPU optimizer offload.
  Those results do not predict the memory or speed of this BF16-trunk integration
  on an H100. SDNQ is not added by this change.
* **ComfyUI:** the custom loader strictly requires architecture metadata and
  FP32 factors, recreates shared conditioning, and preserves original final
  conditioning. These requirements drive the export format here. A standard
  MageFlow loader cannot consume the compressed file directly.

Expanding `head.weight @ down.weight` back to dense projections would allow
ordinary-model loading, but would discard the memory advantage and change the
training parameterization. Running the whole model in FP32 is another possible
baseline, with much greater trunk memory cost. Native mixed-dtype factors are
the practical path for the requested H100 setup. Compression ranks other than
256 are read from metadata; unrelated pruning, SDNQ and other SVD formats are
not implicitly supported.

## Run and validate on the H100

Download the actual repository filename (the card's short example name differs):

```bash
hf download Bluvoll/MageTrail-Shared-AdaLN-SVD \
  transformer/MageTrail-v0.2-rank256-fp64-qr-evaluated.safetensors \
  --revision 23278f33faadf364bd1d5a279e76b457b03957b6 \
  --local-dir /workspace/models/magetrail-compressed
```

Start from [the 20-step smoke config](../examples/mage_flow_compressed_finetune.toml).
Set the dataset and existing MageFlow component paths, then run:

```bash
deepspeed --num_gpus=1 train.py --config examples/mage_flow_compressed_finetune.toml
```

The example deliberately starts at microbatch 1. After a successful run, compare
dense and compressed models using identical captions, resolution buckets,
attention backend, optimizer, checkpoint selection and effective batch. Measure
steady-state images/sec plus peak allocated/reserved memory after optimizer
state allocation. Increase microbatch gradually; adjust accumulation to hold
effective batch constant. Sample fixed prompts/seeds and inspect a longer run
before adopting the result for a full tune. Re-evaluate the learning rate after
changing the parameterization; the example's 2e-6 is a trial setting.

Local numerical checks:

```bash
python -m pytest test/test_mageflow_compression.py test/test_mageflow_training.py test/test_mageflow_execution.py -q
```

They cover expanded-dense equivalence, shared-factor gradients, both checkpoint
variants, reference packed attention, FP32 autocast behavior, device/dtype moves,
optimizer updates, exact FP32 export/reload and invalid-format handling. CPU
reference attention is not a substitute for running real FlashAttention kernels.

An opt-in small CUDA/DeepSpeed integration test exercises mixed-dtype pipeline
steps, accumulation, clipping and checkpoint restore without large model weights:

```bash
RUN_MAGEFLOW_DEEPSPEED_TESTS=1 torchrun --standalone --nproc_per_node=1 \
  -m pytest test/test_mageflow_compression.py -k deepspeed -q
```

This test uses PyTorch AdamW. Also validate the actual `AdamW8bitKahan` preset,
full-model export/reload, production sampling, and interrupt/resume on the H100.
The local environment has CPU-only PyTorch and cannot establish those outcomes.

## Sources

* [Model card and published checkpoint](https://huggingface.co/Bluvoll/MageTrail-Shared-AdaLN-SVD/tree/23278f33faadf364bd1d5a279e76b457b03957b6)
* [Compressed architecture and precision](https://github.com/bluvoll/mage-flow-trainer/blob/b2912d7c4c63cf782d8b9453153db6a24b1bf598/trainer/modeling/compressed_modulation.py)
* [Converter and modulation distillation](https://github.com/bluvoll/mage-flow-trainer/blob/b2912d7c4c63cf782d8b9453153db6a24b1bf598/trainer/tools/distill_modulation.py)
* [Experiments and limitations](https://github.com/bluvoll/mage-flow-trainer/blob/b2912d7c4c63cf782d8b9453153db6a24b1bf598/docs/compressed-modulation.md)
* [ComfyUI loader](https://github.com/bluvoll/ComfyUI-MageFlow-Compressed/blob/d442acfdbc535b34af3f0c5846cf8af5a7d1a0d1/nodes.py)
* [DeepSpeed 0.18.4 engine](https://github.com/deepspeedai/DeepSpeed/blob/v0.18.4/deepspeed/runtime/engine.py)
* [DeepSpeed pipeline engine](https://github.com/deepspeedai/DeepSpeed/blob/v0.18.4/deepspeed/runtime/pipe/engine.py)
