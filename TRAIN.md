# TRAIN.md — train Remy 1.19B from scratch

Reproduce the 1.19B pretraining run: `VeriTransformer(VERI_REMY)` on
Timmy-cut shards with public Terry. Linux + NCCL, 1–4 big GPUs
(B300 288GB reference box, ~126k tok/s combined on 2x).

## What it trains

- 24 layers, dim 2048, 16 q-heads / 4 kv GQA, SwiGLU 5504, RoPE theta 10k,
  per-head qk-norm, differential attention, sliding 2048 on even layers,
  tied 65k embeddings, dropout 0. bf16 compute, fp32 norms/moments.
- 1.19B params. Config lives in `model/model.py` (`VERI_REMY`).

## Data

- Tokenizer: Timmy 65k byte-level BPE. Train your own with
  `python Veri/tokenizer/train.py --vocab 65536 --out timmy.json`
  or use the published `tokenizer/timmy.json`.
- Shards are flat int32 `part_*.npy` files, one doc after another,
  no EOS markers. Point `VERI_TOK_DIRS` at one or more dirs.
- Fresh sequential order, no epochs. Single and multi-GPU runs resume
  each other's checkpoints (consumed counted in packs, same order).

## Batch

- MICRO 1024 x SEQ 1024 per GPU (1.05M tok/step/GPU), rank-strided:
  rank r eats every WORLD-th pack starting at r.
- Dollars-per-token is flat across 1/2/4 GPUs, wall-clock isn't.

## Optimizer (public Terry, `training/terry.py`)

- Factored adaptive tables, grafted matrices, table nesterov.
  One momentum state per parameter.
- WSD: flat LR (we use 0.003). Do NOT use an expiring schedule — a
  cosine that expired mid-run cost ~1000 flat steps before anyone
  caught it by math. Schedules that expire are banned.
- Warmup 100 (rms + nesterov + ns depth ramp). Grad clip 1.0.

## Run it

```bash
# 2 GPUs, 400 steps from step 0
VERI_TOK_DIRS=/data/shards VERI_CKPT_DIR=/data/ckpt \
torchrun --nproc_per_node=2 Veri/training/train.py

# resume from the latest checkpoint in the ckpt dir
VERI_RESUME=1 VERI_TOK_DIRS=/data/shards VERI_CKPT_DIR=/data/ckpt \
torchrun --nproc_per_node=2 Veri/training/train.py
```

- `VERI_STEPS` (default 10): steps this fire. `VERI_SAVE_EVERY`
  (default 9): checkpoint cadence — weights + optimizer + step +
  data position + RNG states, bit-identical resume.
- Fixed 3-prompt sampler every 10 steps, so quality is visible live
  in the training log. Rank 0 owns ckpt/query/logs.
- Fused swiglu/rope/rmsnorm graft in automatically on CUDA.

## Gotchas

- Outputs are the dashboard, loss is the smoke alarm. If loss looks
  impossibly good while samples babble, the metric is lying — believe
  the samples and go read the attention code.
- Greedy samples lie about voice, template mismatch lies about SFT.
  Keep eval prompts in-template from day one.
