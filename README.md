# Veri

From-scratch 1.19B transformer with custom tokenizer and benched architecture. Tokenizer ties Tiktoken at 3x smaller vocab. Optimizer beats Muon with a private variant at >15x less states.

- **Veri 1.19B**: 24 layers, dim 2048, 16Q/4KV GQA, SwiGLU, RMSNorm, RoPE, QK-Norm, differential attention, sliding 2048 interleaved, tied embeddings. Full config in `TRAIN.md`.
- **Timmy**: 65k BPE, byte level, single-digit numbers, think tags atomic. `tokenizer/timmy.json` 5.1MB.
- **Pretrain**: 20B tokens and falling (loss 11 → 5.1), single → dual B300 mid-campaign, chunked CE, checkpointed blocks.
- **Weights**: [veriepic/Veri-Base on Hugging Face](https://huggingface.co/veriepic/Veri-Base) — transformer + gguf + full checkpoint, mirrored in `weights/`.

Public optimizer baseline in this repo is Muon. A private optimizer variant with the same loss at >15x less states exists and is withheld. DM for file.

## Zoo results (118M, 1000 steps, one harness)

Full table in `results.json`. 35 archs, one fixed run — our variants plus faithful cuts of 2026 frontier models. Same budget, same batches, same optimizer.

| rank | arch | loss | whose |
|---|---|---|---|
| 1 | qk-post | 2.27 | ours |
| 2 | remy-post | 2.28 | ours |
| 2 | remy-sand | 2.28 | ours |
| 4 | remy-ls | 2.34 | ours |
| 5 | qk-ls | 2.35 | ours |
| 6 | gemma-4 | 2.38 | Google |
| 7 | parallel-qk | 2.46 | ours |
| 8 | mla-lite | 2.48 | ours |
| 8 | remy-parallel | 2.48 | ours |
| 10 | parallel | 2.49 | ours |

Rest of field: gated 2.50, reglu 2.59, looped-act 2.60, mqa 2.62, wide-shallow 2.62, liquid-conv (Liquid) 2.64, combo 2.65, remy-mtp 2.65, inkling (Thinking Machines) 2.67, geglu 2.68, mha8 2.68, narrow-deep 2.71, qknorm 2.73, remy 2.73, layerdrop 2.78, step-5 (StepFun) 2.84, glm-53 (Zhipu) 2.87, granite-40 (IBM) 2.93, oss-120 (OpenAI) 2.96, kimi-k3 (Moonshot) 2.97, base 3.07, nemotron-mamba (NVIDIA) 3.13, remy-grind 3.49, qwen35 (Qwen) 3.53. ds-v41 (DeepSeek) diverged.

The zoo lives in `training/benchmark/arch.py` (builders + honest deviations). The runner is `training/benchmark/bench.py`.

```bash
python Veri/tokenizer/test_timmy.py
python Veri/training/bench_opt.py --steps 200
```
