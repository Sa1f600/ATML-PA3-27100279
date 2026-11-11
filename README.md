# ATML Programming Assignment 3

**Task 1 subtasks 1-7 are implemented:** training, benchmark evaluation, numerical
diagnostics/plots, and qualitative generation plus human-annotation tooling.
No full training or benchmark experiment has been run; no measured benchmark
scores or qualitative labels are supplied.

## Task 2: Colab setup

Task 2 uses the original Qwen3-0.6B, a frozen `google/siglip-base-patch16-224`
encoder, a FP32 768-to-1024 GELU connector, and rank-16 attention LoRA enabled
only in stages 2-3. Download COCO 2014 images/annotations and VQAv2 question and
annotation JSON files into the paths in `configs/task2/vlm_196.json`, then run:

```bash
python -m task2_vlm.prepare
python -m task2_vlm.prepare --features-only --cache-features
python -m task2_vlm.preflight --require-features
python -m task2_vlm.smoke --output results/task2/smoke_test.json
python -m task2_vlm.train
```

The manifest builder deterministically selects the specified caption, natural
VQAv2, and annotation-derived pair sizes and stores image/question/pair IDs.
Feature caching freezes SigLIP after writing FP16 `[196, 768]` features, so Qwen
training never keeps the encoder on GPU. Training writes recovery checkpoints
every 250 main-run updates; resume with `--resume checkpoints/task2/vlm_196/update_XXXX`.
Run the isolated GPU smoke test after preflight and before the full curriculum:
it performs one real connector-only caption update and one real connector-plus-LoRA
grounding update, writing only the requested standalone JSON result (never a
training checkpoint).
Evaluate a stage-end or final checkpoint with the four prescribed image conditions:

```bash
python -m task2_vlm.evaluate --checkpoint checkpoints/task2/vlm_196/update_4500 --split final
```

## Task 3: test-time compute

Task 3 is fully separate from Tasks 1 and 2. Its offline component reads the
course-supplied `ATML-PA3-Data` rollout tables; it does not require any trained
model or results from the earlier tasks. Put that unzipped bundle outside Git,
then update the two source paths in `configs/task3/offline.json` if its folder
names differ from the template.

```bash
python -m task3_test_time.offline --config configs/task3/offline.json
python -m task3_test_time.analyze --offline results/task3/offline
```

The offline run writes the seed-6304 development/held-out split IDs and all ten
nested rollout orders before scoring. It saves per-question selection records,
adaptive-policy choices, paired bootstrap intervals, signal summaries, and plot
inputs for both exports. Selection and stopping never read gold answers.

The 50-question MATH live experiment is separate and resumable. First run the
offline analysis; then point `--manifest` to its MATH split manifest and point
`--questions` to the course bundle's matching sample-id/question table:

```bash
python -m task3_test_time.live \
  --manifest results/task3/offline/math/MATH_split_and_orders.json \
  --questions data/ATML-PA3-Data/math_questions.parquet \
  --output results/task3/live_math
```

Each completed question is synchronously appended to `records.jsonl`; rerun the
same command after a timeout and it skips those questions. This makes the only
new-generation part safe to run in a temporary notebook session.

## Environment

Use Python 3.10-3.12 on a machine with an NVIDIA CUDA GPU. Install an appropriate
CUDA build of PyTorch 2.8.0 for that machine, then install the remaining pinned
dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

PyTorch 2.8.0 is paired with Torchvision 0.23.0. Kaggle's preinstalled
Torchvision may belong to a different PyTorch version; leaving it in place can
make the evaluation-harness import fail with `operator torchvision::nms does
not exist`. Install the pinned pair through the requirements above. To repair
an existing installation of this project's PyTorch, use
`python -m pip install --upgrade torchvision==0.23.0`, then retry in a new
Python subprocess. Prepared datasets and checkpoints are unaffected.

The prescribed full run deliberately rejects CPU and Apple MPS execution: it
uses CUDA FP16 autocast, dynamic gradient scaling, and FP32 model/optimizer
parameters. It does not quantize the model. Available GPU memory depends on
sequence lengths and attention kernels; the code does not silently reduce the
required batch or sequence length to fit a device. Host RAM must also hold a
pristine FP32 parameter reference for displacement measurements.

Run all commands below from the repository root. The initial preparation/model
load requires internet access to Hugging Face. No credentials are needed for
these public repositories.

Before a long training or evaluation run on Kaggle, use the preflight check:

```bash
python -m task1_finetuning.preflight --benchmarks --checkpoints
```

It checks CUDA and pinned package versions, verifies all prepared-data hashes,
loads each pinned benchmark to confirm its expected full split size, and checks
the layout of any existing recovery checkpoints. It does not train or generate
benchmark answers.

## Task 1.1: prepare the shared data once

```bash
python -m task1_finetuning.prepare_data --config configs/task1/full_ft.json
```

Preparation resolves model and dataset revisions to immutable Hugging Face
commit IDs. It scans `meta-math/MetaMathQA` in source order, selecting the first
40,000 eligible examples. It excludes empty queries/responses, normalized
query/original-question matches to GSM8K test questions, and full formatted chats
over 2,048 tokens. Answers are never truncated and selected rows are never
shuffled.

Normalization is Unicode NFKC, case folding, and collapsed whitespace, preserving
mathematical punctuation. The native Qwen chat template is called with
`enable_thinking=False`; its fixed empty thinking block belongs to the masked
prompt. Only assistant content and the end-of-turn token are supervised. The
trailing chat-format newline is masked but counts toward the formatted length.
No system message is added.

Prepared token IDs, labels, tokenizer files, hashes, and resolved revisions live
in `cached/task1/prepared/`. Training verifies all cached file hashes before use.
The selected source indices/question-response hashes and the fixed activation
probe IDs are copied to `manifests/task1/` for version control. Probes contain
32 GSM8K test questions and 32 HellaSwag validation contexts, sampled without
replacement with NumPy `default_rng(6304)` before model results are available.

Preparation refuses to replace an existing cache or existing data-selection manifests.
Reuse the prepared files across later methods so their examples and tokenization
stay identical. Change paths explicitly if intentionally preparing a separate run.

## Task 1.1: train

```bash
python -m task1_finetuning.train_full_ft --config configs/task1/full_ft.json
```

The JSON configuration encodes the assignment's fixed recipe: seed 6304,
40,000 examples, one epoch, microbatch 2, accumulation 16, 1,250 successful
optimizer updates, AdamW, peak LR 2e-5, 38 warmup updates, cosine decay,
weight decay 0.01 except biases/parameters with fewer than two dimensions,
gradient norm clipping at 1.0 after unscaling, and gradient checkpointing.
All original parameters train, with no starting-point penalty (lambda = 0).

For each effective batch, sum response-token cross-entropies and divide by the
total number of valid response tokens in all 32 examples. Each microbatch uses
this same denominator; do not divide again by 16 or average microbatch means.

The LR rises linearly from peak/38 at update 1 to peak at update 38, then follows
a cosine to zero at update 1,250. This states the otherwise unspecified schedule
endpoint convention explicitly. TF32 is disabled and deterministic algorithms
are requested with warnings for unsupported operations.

If AMP skips an optimizer step because of overflow, log the skipped attempt,
reduce the scale, restore that effective batch's RNG state, and retry the same
examples. Do not advance the data position or LR. This preserves both the 40,000
unique-example budget and 1,250 successful updates; retry presentations and
their time are recorded separately. Repeated overflow aborts rather than
silently producing a shorter training run.

## Outputs and recovery

Recovery checkpoints are saved at updates **312, 625, 937, 1250**, corresponding
to **9,984, 20,000, 29,984, 40,000** examples:

```text
checkpoints/task1/full_ft/update_0625/
    config.json
    model.safetensors          # or sharded safetensors files/index
    tokenizer files
    trainer_state.pt           # optimizer, scheduler position, scaler, RNG, data position
    run_config.json
    checkpoint_summary.json
```

Resume from the latest complete checkpoint:

```bash
python -m task1_finetuning.train_full_ft \
  --config configs/task1/full_ft.json \
  --resume checkpoints/task1/full_ft/update_0625
```

Resume validates configuration/data identity. It keeps abandoned post-checkpoint
log records in timestamped audit files and resumes the active logs from the saved
position. A `.saving` directory is an incomplete checkpoint, never a valid
resume source; inspect/remove an abandoned `.saving` directory before retrying
that checkpoint write. Only load trusted `trainer_state.pt` files created by
your own runs.

Results are written under `results/task1/full_ft/`:

- `training.jsonl`: response-token SFT loss, LR, response-token count, gradient
  norm, AMP scale, skipped attempts, examples seen, and timing per update.
- `resources.json`: trainable parameter count, training time, diagnostic and
  checkpoint overhead, peak allocated/reserved GPU memory, CPU reference storage,
  and extra presentations caused by overflow retries.
- `checkpoints.jsonl`: model/recovery checkpoint sizes and save time.
- `displacement_0625.json`, `displacement_1250.json`: full-model L2 distance from
  starting parameters, counting tied parameters once.
- `run_config.json`, `environment.json`: resolved model revision, data hash,
  actual package versions, GPU/CUDA details, and Git commit if available.

The CPU reference is used only for the required displacement diagnostic. It is
not old-task training data and is never used in the Full FT objective.

## Task 1.2: Full Fine-Tuning with L2-SP

Reuse the exact same prepared dataset, selected IDs, tokenizer and original
model revision as Full FT. If preparation has not been run, run the preparation
command above once. Then start the independent regularized run:

```bash
python -m task1_finetuning.train_l2_sp
```

Configuration: `configs/task1/full_ft_l2_sp.json`. Its training settings match
Full FT, with the manual's additional objective:

```text
Omega = (1/2) * sum over UNIQUE original parameters of (theta - theta0)^2
total_loss = response_token_averaged_SFT_loss + 0.001 * Omega
```

The sum includes biases and normalization parameters and counts tied weights
once. It is not averaged over parameters. AdamW weight decay remains 0.01 with
the existing exclusions; L2-SP is an additional starting-point penalty.

The reference is an independent frozen FP32 CPU copy of the original model,
created before any resumed checkpoint is loaded. On resume it is reconstructed
from the same immutable original revision, never from the trained checkpoint.
No checkpoint from the unregularized run is used to initialize L2-SP.

To bound GPU memory, copy reference weights to the GPU in chunks of at most
1,048,576 FP32 elements (4 MiB). Add the analytic penalty gradient,
`lambda * (theta - theta0)`, once per effective batch, scaled by the current AMP
scale. This is mathematically the same gradient as backpropagating the penalty.
The combined SFT and penalty gradients are then unscaled, checked for overflow,
and clipped together before AdamW. FP32 differences and FP64 reductions are used
for the logged penalty. This keeps the penalty outside FP16 autocast.

This CPU reference policy trades GPU memory for host-to-device transfers. Per
attempt, `training.jsonl` additionally records `omega`, `lambda_omega`,
`total_loss`, `l2_sp_lambda`, `regularization_seconds`,
`reference_transfer_seconds`, `reference_transfer_bytes`, and
`largest_reference_gpu_chunk_bytes`. Losses refer to parameters before that
attempt's optimizer step. Transfer times use CUDA events around each reference
copy; overall regularization time also includes computation, gradient addition,
allocation and synchronization. It is already included in training time, so do
not add it a second time when comparing methods.

`resources.json` records cumulative regularization time, reference transfer
bytes/time (including overflow retries), CPU reference bytes, reference-copy
setup time, and the largest transient GPU reference chunk. Memory peaks include
regularization. Reference-copy setup is outside the measured training loop.

Outputs are separate:

```text
checkpoints/task1/full_ft_l2_sp/update_0312/
checkpoints/task1/full_ft_l2_sp/update_0625/
checkpoints/task1/full_ft_l2_sp/update_0937/
checkpoints/task1/full_ft_l2_sp/update_1250/
results/task1/full_ft_l2_sp/training.jsonl
results/task1/full_ft_l2_sp/resources.json
results/task1/full_ft_l2_sp/displacement_1250.json
```

All four checkpoints support recovery. Only the final L2-SP checkpoint is marked
for evaluation, and full-model displacement is measured at final, as required.
Resume the latest complete checkpoint using:

```bash
python -m task1_finetuning.train_l2_sp \
  --resume checkpoints/task1/full_ft_l2_sp/update_0625
```

## Task 1.3: LoRA at ranks 4 and 16

Reuse the shared prepared dataset. Start these as two independent processes,
each loading the original pinned Qwen checkpoint:

```bash
python -m task1_finetuning.train_lora --rank 4
python -m task1_finetuning.train_lora --rank 16
```

Configurations are `configs/task1/lora_4.json` and `configs/task1/lora_16.json`.
Each uses the same 40,000 examples in source order, microbatch 2, accumulation
16, 1,250 updates, checkpoint schedule, token-normalized SFT objective and AMP
handling as Full FT. The LoRA peak LR is 2e-4, rank/alpha pairs are (4,8) and
(16,32), and lambda is zero.

`task1_finetuning/lora.py` wraps exactly `q_proj`, `k_proj`, `v_proj`, `o_proj`,
`gate_proj`, `up_proj`, and `down_proj` in every decoder layer. It verifies the
complete target set before training. The original backbone is frozen FP16;
adapter A/B parameters are FP32, with FP16 linear computation under CUDA
autocast and FP32 optimizer states. No embedding, normalization, LM-head, or
bias parameters are trained. Non-reentrant gradient checkpointing permits
adapter training despite frozen input embeddings.

The adapter computes `(alpha/r) * B * A * dropout(x)`. Dropout is applied only
on the adapter input, with probability 0.05. A uses Kaiming-uniform
initialization with `a=sqrt(5)`; B starts at zero. Consequently initial logits
match the frozen original model. The adapter initialization seed is reset to
6304 immediately before installing adapters, so LoRA-16 initialization can be
reproduced for OPLoRA later. This does not make rank-4 and rank-16 factors equal;
they have different shapes.

For each rank, `results/task1/lora_<rank>/adapter_parameters.json` records the
actual adapter parameter count, frozen parameter count, exact targeted module
paths, dropout, scaling and initialization. For the current Qwen3-0.6B
architecture there are 196 targeted projections, with **2,523,136** trainable
parameters at rank 4 and **10,092,544** at rank 16. The latter is exactly four
times the former. These counts were checked from model shapes without loading
pretrained weights; training records its own measured counts.

The usual training/resource/checkpoint logs are written to separate
`results/task1/lora_4/` and `results/task1/lora_16/` folders. Resources include
adapter setup time. No full starting-weight CPU reference is retained for these
runs because they need neither L2-SP nor full-model displacement measurement.

Adapter-only recovery checkpoints contain:

```text
checkpoints/task1/lora_4/update_0625/
    adapter_model.safetensors
    adapter_config.json         # original model/revision, rank, alpha, targets
    tokenizer files
    trainer_state.pt            # optimizer, LR position, scaler, RNG, data position
    run_config.json
    checkpoint_summary.json
```

This repository uses its own small `atml_lora_v1` checkpoint format, not PEFT's
format. `inject_lora` and `load_adapter` reconstruct it on a fresh original
backbone. Backbone weights are downloaded from the pinned original revision
rather than redundantly saved with every adapter checkpoint. All four
checkpoints support recovery; only the final checkpoint requires evaluation.

For example, resume rank 16 with:

```bash
python -m task1_finetuning.train_lora --rank 16 \
  --resume checkpoints/task1/lora_16/update_0625
```

Benchmark evaluation, adapter merging, and the required merged/unmerged logit
comparison are implemented in subtask 5 below.

## Task 1.4: OPLoRA

After preparing the shared data, run:

```bash
python -m task1_finetuning.train_oplora
```

Configuration: `configs/task1/oplora_16.json`. Rank 16, alpha 32, peak LR 2e-4,
dropout 0.05, target modules, optimizer, examples/order, tokenization, AMP and
checkpoint settings match LoRA-16. Protection rank is 128. There are still
10,092,544 trainable adapter parameters; the singular bases are frozen buffers.

On first launch, the code loads the original pinned checkpoint in FP32 and
computes SVDs of all 196 target weight matrices. It does not first round the
weights to FP16 and cast them back. It uses `torch.linalg.svd` with
`full_matrices=False` (CUDA driver `gesvd`), retaining the leading 128 left and
right singular vectors in FP32. No randomized/approximate SVD is used. SVD runs
on the CUDA GPU when available; CPU is supported by the basis helper for local
numerical checks. The main training entry point still requires CUDA.

The bases and metadata are cached in `cached/task1/oplora_bases/`. Preparation
prints progress once per target matrix and can take appreciable time. The
original FP32 preprocessing model is then released. Subsequent launches reuse
the cache after validating the original model revision, protection rank,
targets, dtype and basis-file checksum. Incomplete `.preparing` directories are
never treated as valid caches.

Adapters are initialized with exactly the same code, order and seed as LoRA-16.
The protection wrappers reuse those A/B tensors without random draws. During
each forward pass they compute:

```text
B_effective = B - U_k @ (U_k.T @ B)
A_effective = A - (A @ V_k) @ V_k.T
correction = (alpha/r) * B_effective @ A_effective
```

Projection math runs in FP32 outside autocast, keeping gradients through both
effective factors. The adapter linear operations use FP16 autocast as in LoRA.
No dense square projection matrices are formed; neither the frozen backbone
nor bases receive gradients. Orthogonality holds to floating-point tolerance;
the FP16 forward pass also introduces rounding.

Recovery checkpoints live in `checkpoints/task1/oplora_16/update_<update>/`:

```text
adapter_model.safetensors       # raw, unprojected trainable A/B
adapter_config.json             # atml_oplora_v1, protection rank, basis checksum
bases.safetensors               # exact frozen FP32 U_k/V_k for recovery
basis_metadata.json
inference_adapter/
    adapter_model.safetensors   # projected A_effective/B_effective
    adapter_config.json        # ordinary atml_lora_v1, marked inference-only
trainer_state.pt                # optimizer, scheduler position, scaler, RNG, data position
tokenizer files
run_config.json
checkpoint_summary.json
```

Each recovery checkpoint carries its own bases so it can resume in a new Kaggle
session without the original cache. It reconstructs the protected model and
restores raw factors and optimizer state. Do not resume training from
`inference_adapter/`: that export contains projected factors for evaluation,
not the original optimization state. The ordinary LoRA loader rejects raw
OPLoRA checkpoints, preventing accidental inference with unprojected factors.

```bash
python -m task1_finetuning.train_oplora \
  --resume checkpoints/task1/oplora_16/update_0625
```

Every checkpoint exports projected FP32 factors. These can be loaded using the
existing ordinary LoRA wrappers on the original backbone, without bases or
projection operations. Subtask 5 will merge these projected factors into fresh
backbone copies for benchmark evaluation.

Alongside common logs, `basis_resources.json` records basis tensor/file bytes,
the checksum, SVD device/version, preprocessing seconds and preprocessing GPU
peak. `resources.json` records basis storage, preprocessing time, cumulative
cache setup time, and projected-factor export time/bytes. Export time covers
factor projection, CPU transfer and writing; total checkpoint time additionally
includes raw states and basis copies. Export time is part of checkpoint time,
so do not add it twice. Recovery sizes include every copied basis and inference
export, while `model_bytes` counts only root model/adapter tensors.

## Task 1.5: benchmark evaluation

Install the updated requirements before evaluation. `lm_eval==0.4.9.1` requires
Datasets below version 4, so this repository now pins `datasets==3.6.0` together
with compatible Transformers, Accelerate and PEFT versions. The custom training
adapters still use this repository's PyTorch implementation, not PEFT.

Evaluate the original model once after data preparation (it does not require
training checkpoints):

```bash
python -m task1_finetuning.evaluate --condition baseline
```

After the training runs, evaluate all required conditions:

```bash
python -m task1_finetuning.evaluate
```

The fixed condition names are `baseline`, `full_ft_midpoint`, `full_ft_final`,
`l2_sp_final`, `lora4_final`, `lora16_final`, and `oplora_final`. Use
`--condition <name>` to evaluate one condition. The midpoint is update 625; final
is update 1250. Quarter checkpoints and the L2-SP midpoint are recovery-only.
Each finished benchmark task is saved separately. If evaluation is interrupted,
rerun the same command: it validates and reuses each complete task, then runs
only unfinished or incomplete tasks. A condition is fully complete only after
its `results/task1/evaluation/<condition>/complete.json` file is present.
While a benchmark is running, the evaluator prints progress at roughly 5% intervals;
GSM8K reports completed generations and the multiple-choice benchmarks report
completed likelihood requests.

`configs/task1/evaluation.json` specifies the paths and numerical tolerances.
All benchmark models have FP16 weights and dropout disabled. GSM8K uses the
stock `gsm8k_cot` task, zero-shot native non-thinking chat, greedy generation,
1,024 new tokens, and its original stopping rules and answer filters. HellaSwag,
BoolQ and ARC-Easy use their stock zero-shot completion prompts and candidate
likelihood scoring, without a chat template. Complete split sizes are checked:
1,319 / 10,042 / 3,270 / 2,376. Evaluation offers no subset limit option.

`manifests/task1/benchmark_lock.json` fixes the harness version, stock task
definition hashes, and dataset revisions. GSM8K/HellaSwag revisions match the
shared preparation/probe manifests. Canonical dataset aliases are used to load
the same stock datasets at these immutable revisions. Prompt definitions,
choices, metrics, answer filters and stopping rules are retained unchanged.

Adapters are loaded on an original backbone for the unmerged check, then merged
into a separately loaded fresh original FP16 backbone using FP32 corrections.
OPLoRA always uses its exported projected factors. The check compares logits
at the last token of two fixed small prompts with the same tokenizer/padding.
FP16 branch addition and merged-weight rounding can differ numerically: the
explicit tolerances are absolute 0.1, relative 0.01, and relative L2 error at
most 0.01. Check results are saved and evaluation aborts if they fail. These are
verification tolerances, not task hyperparameters selected from benchmarks.

Each `results/task1/evaluation/<condition>/` directory contains task result
JSON, raw per-example JSONL, `aggregate.json`, and an input/model contract. GSM8K
records preserve both stock filters, extracted answers, generated token IDs,
token counts, and cap/termination/truncation flags. Missing extractions remain
incorrect under stock grading. A response ending with EOS or a stock stop
sequence exactly at the cap is not flagged truncated. Generation batch size is
1 so flags correspond to actual generation rather than re-tokenized text.

Complete evaluations are reused only when the model tensor checksums and
settings match. Interrupted evaluations can be rerun; incomplete files are
replaced and the completion marker is written only after all four benchmarks.
New training checkpoints identify the prepared-data checksum in their run
configuration, preventing evaluation of a checkpoint from a different selection.

## Task 1.6: diagnostics and plots

After all five training runs (including OPLoRA), run:

```bash
python -m task1_finetuning.diagnostics
python -m task1_finetuning.analyze
```

Diagnostics cover the six nonbaseline conditions at `down_proj` layers 0, 13,
27, with k=128. The exact bases saved in the OPLoRA checkpoint are reused for
every method. Full FT updates are trained FP32 weights minus original FP32
weights. Adapter updates use their FP32 scaled factors, projected for OPLoRA,
without the additional rounding introduced by merging into FP16 weights.

The original FP32 model runs on the fixed 32 GSM8K and 32 HellaSwag probe
prompts, using the same stock benchmark formatting. Forward pre-hooks cache
`down_proj` inputs at the last nonpadding prompt token, with no answer
continuation. Every method applies its local weight change to these SAME
cached inputs. The cache checks original revision, probe IDs, basis checksum,
task definitions and tensor checksum. All diagnostic calculations use FP32.

Per-condition JSON files under `results/task1/diagnostics/` contain rho_k, eta,
and all 64 per-probe R(z) values for each layer. When an update is exactly zero,
rho_k is undefined and saved as null; eta and R(z) are zero. These local
measurements do not establish why an end-to-end behavioral change occurred.

`analyze` requires completed benchmark scores, all training logs and diagnostic
outputs. It produces CSV tables in `results/task1/analysis/` for benchmark scores
and baseline changes in percentage points, the five requested controlled
comparisons (including Full FT progression), resources/checkpoint sizes,
parameter displacement, weight diagnostics and math/non-math activation
summaries. It excludes overflow attempts from successful-update learning curves.

Four figures are saved as PNG and PDF under `figures/task1/`: combined SFT loss,
L2-SP objective components, benchmark scores, and weight/activation diagnostics.
No findings or report paragraphs are written automatically.

## Task 1.7: qualitative generation and annotation

The user-supplied source is tracked verbatim at
`manifests/task1/first_plot_questions.yaml`, pinned by GitHub revision and SHA256
in `qualitative_prompt_source.json`. Its eight plain questions and matching
JSON/template variants are preserved, including JSON system messages. Upstream
judge prompts and its 100-sample setting are ignored: the assignment specifies
one greedy response, and no external judge is called.

```bash
python -m task1_finetuning.qualitative
```

This runs all seven conditions with thinking disabled and a 512-new-token cap.
Use `--condition <name>` to generate one condition at a time. Per-condition JSON
records messages, rendered prompts, responses, input/output token IDs,
truncation flags, model checksums and the source prompt checksum. All 168
responses are grouped by underlying question, format and condition in
`results/task1/qualitative/responses_by_question.json`.

The first and last plain-order question IDs are `what_is_your_wish` and
`enough_of_my_husband`. Once all responses exist, `annotations.csv` contains
their fixed 42-response subset, with blank columns for your manual labels:

- format compliance: pass / partial / fail / not_applicable;
- coherence/relevance: good / mixed / poor;
- repetition: none / some / severe;
- unnecessary mathematical framing: yes / no;
- inappropriate content: yes / no / uncertain;
- short supporting excerpt and notes.

Complete these labels yourself, preserving response text and truncation flags.
Flagged observations require an exact response excerpt of at most 300
characters. Then validate and count labels using:

```bash
python -m task1_finetuning.qualitative \
  --summarize-annotations results/task1/qualitative/annotations.csv
```

The summary contains label counts by condition; interpretation and report
writing remain your work. An existing annotation file is never overwritten.

## Verification

```bash
python -m pytest -q
```

Run the individual subtask verification suites with:

```bash
python -m pytest -q tests/test_evaluation.py
python -m pytest -q tests/test_diagnostics.py
python -m pytest -q tests/test_qualitative.py
```

The new suites verify merging and logit failure detection, the real pinned
harness on a tiny offline synthetic task, stock extraction/stopping definitions,
termination metadata, numerical diagnostic formulas, last-token activation
hooks, projected OPLoRA updates, percentage-point calculations, all plot/table
outputs, the actual supplied YAML, 168-response grouping, the fixed annotation
subset, and manual-label validation. Synthetic test scores are temporary test
fixtures and are not assignment results. Full GPU experiments remain unrun.

CPU tests check token-weighted accumulation against full-batch gradients on a
tiny randomly initialized Qwen3, prompt/padding masking, contamination filtering,
the LR/checkpoint schedule, optimizer grouping, tied-parameter displacement,
RNG recovery, and model/optimizer checkpoint continuation. If Qwen tokenizer
files exist under `cached/huggingface/`, an additional test checks the real chat
template and supervised EOT token. This does not require pretrained weights.

L2-SP tests additionally compare the analytic scaled penalty gradient against
direct autograd of the full objective, including unequal response lengths,
clipping and an optimizer step. They check inclusion of biases/norms, tied
parameter deduplication, reference immutability, zero initial penalty, and
matching training configurations.

LoRA tests check the forward equation, zero-update initial logits, all seven
targets per layer, FP16 frozen backbone/FP32 adapters, fourfold parameter scaling,
reproducible initialization, adapter-only checkpoints, and exact continuation
with restored adapter weights, optimizer state and dropout RNG. A tiny Qwen3
training check uses non-reentrant checkpointing and CPU FP16 autocast; full CUDA
training still requires verification on the target GPU.

OPLoRA tests compare effective factors with explicit small projection matrices,
check both orthogonality constraints and preservation of selected singular
triples, compare gradients, confirm FP32 projection inside autocast, verify
identical unprojected initialization to LoRA-16, and test cache integrity.
Projected exports reproduce protected-model logits without bases, and recovery
tests restore raw factors/optimizer/RNG using only checkpoint-contained bases.

## Attribution

Implementation authored with OpenAI Codex coding assistance. It uses public APIs
from [PyTorch](https://docs.pytorch.org/docs/stable/notes/amp_examples.html),
[Hugging Face Transformers](https://huggingface.co/docs/transformers/v4.53.1/en/index),
and [Hugging Face Datasets](https://huggingface.co/docs/datasets/).
Model/template: [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B).
Data: [MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA),
[GSM8K](https://huggingface.co/datasets/openai/gsm8k), and
[HellaSwag](https://huggingface.co/datasets/Rowan/hellaswag).
Evaluation uses [EleutherAI's lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).
Qualitative prompts are from
[Emergent Misalignment](https://github.com/emergent-misalignment/emergent-misalignment/blob/e40daef5791139a2417e22034ab0a0e72900524b/evaluation/first_plot_questions.yaml).
No external training implementation has been copied into the repository.
