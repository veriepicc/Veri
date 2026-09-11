# Veri benchmarks

*RTX 3060 laptop 6GB and H100. Torch 2.11. Same init, same batches, same seeds. No cherry picking.*

## timmy vs tiktoken

Byte level BPE, 65,536 vocab. Trained on 500k fineweb-edu docs plus 153k Veri SFT texts, 19.9 min on a CPU box. Tokens per string, lower is better.

| test            | timmy 64k | o200k | cl100k |
|-----------------|----------:|------:|-------:|
| plain english   |        14 |    16 |     16 |
| veri chat       |        23 |    29 |     29 |
| reasoning trace |        22 |    29 |     29 |
| code            |        61 |    56 |     56 |
| math            |        38 |    27 |     27 |
| unicode         |        13 |    11 |     12 |
| **TOTAL**       |   **171** |   168 |    170 |
| chars per token |      3.11 |  3.16 |  3.14 |

Wins plain english, chat and reasoning outright. Code gap shrunk 72 to 61 with real data. Math gap is deliberate: single digit numbers split for arithmetic, trades tokens for reasoning. 171 vs 168 at 3x smaller vocab. 357k tok/s encode. Every think tag one token. Six exact roundtrips.

File: `Veri/tokenizer/timmy.json`, 4.8 MB.

## architecture

Five variants, 11 to 12M params, same init, same 200 TinyStories batches, same optimizer.

| step | base | qknorm | sliding | theta | deep | combo |
|------|-----:|-------:|--------:|------:|-----:|------:|
| 1    |10.823|  10.824|  10.823 | 10.823| 10.817| 10.824|
| 25   | 5.738|   6.145|   5.769 |  5.733|  6.152|  6.130|
| 50   | 4.898|   4.578|   4.918 |  4.932|  4.863|  4.590|
| 100  | 3.387|   2.819|   3.305 |  3.366|  3.835|  2.830|
| 200  | 2.421|   2.239|   2.404 |  2.459|  3.686|  2.257|

QK-Norm wins at 2.24. Slow starter, fastest finisher. QK-Norm plus sliding ties at 2.26 while halving attention on even layers — kept. High rope theta hurts at short context, expected. Deep thin flops, width wins when params are scarce.

Locked 1.19B: 24 layers, dim 2048, 16Q/4KV GQA, SwiGLU, RMSNorm, RoPE theta 10k, QK-Norm, sliding 2048 interleaved, tied embeddings, Timmy 64k, 4k context.

## optimizer

Terry: factored adaptive tables, grafted orthogonalized matrices, table nesterov. One momentum state per parameter. Muon is the faithful repro.

7M, 1000 steps, same init (10.8491). Full 41-point curves: `Veri/training/benchmark/duel_7m.json`.

| optimizer | final   | vs terry |
|-----------|--------:|---------:|
| Terry     |  2.9289 |        — |
| Muon      |  3.2350 |     +9.5% |
| AdamW     |  3.5257 |    +16.9% |

Terry below both at nearly every checkpoint (muon edges step 25).

300-step ablations: tb08 3.8429 wins. Sign-graft flops (5.02), no-tnest 4.01, graft-l 3.86.

States on a 140M transformer, same weights for all:

| optimizer | states   | peak VRAM |
|-----------|---------:|----------:|
| AdamW     | 1125.3MB |     2.85GB |
| Muon      |  562.7MB |     1.90GB |
| Terry     |  562.6MB |     1.90GB |
| Terry private | 74.7MB |     1.45GB |

Private variant: 15x less than AdamW, no loss difference (3.91 vs 3.92 transformer bench, 0.0000 vs 0.0000 toy). How it saves is withheld.

## toy

Small regression, 300 steps, cosine scheduled.

| optimizer | lr    | final loss |
|-----------|-------|-----------:|
| Terry     | 0.005 |     0.0245 |
| Terry     | 0.02  |     0.0001 |
| AdamW     | 3e-4  |     0.3090 |
| AdamW     | 1e-3  |     0.0616 |

## 140M duel, H100

113M transformer (12 layers, dim 768, bf16), 400 steps, batch 8 seq 128, same packs replayed for all. 15 lanes: 14 terry configs + muon bar. Full curves: `Veri/training/benchmark/duel_140m.json`.

| lane | final  |
|------|-------:|
| ns7 | 3.9844 |
| lr003 / rms010 / rms020 / gentle2 | 4.00 |
| base / b85 / b95 | 4.03 |
| hot / lr008 | 4.06 |
| muon-bar | 4.0938 |
| no-graft / tb2-90 | 4.13 |
| no-tnest / sign-tnest | 4.16 |

11 of 14 beat muon. The 3 that don't: two deliberate ablations plus one bad beta. No explosions at any lr 0.003–0.008.

Earlier run spiked (11 to 13 at step 50, loss-46 blowup at lr 0.02). Cause: full lr on noise momentum. Fix: internal warmup ramps (RMS, nesterov blend, NS depth 0.2x to 1.0x over 100 steps) plus external warmup 100. Gone. Retired scripts not shipped — the live duel runs with `Veri/training/bench_opt.py`.
