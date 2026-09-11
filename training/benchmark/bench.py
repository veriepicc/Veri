import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arch import BUILDERS, DESCRIBE, count_params, flops_per_token, kv_bytes_per_token
from terry import Terry, cosine_schedule


def load_batches(n_batches, batch, seq):
    import tiktoken
    from datasets import load_dataset

    enc = tiktoken.get_encoding("gpt2")
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    need = n_batches * batch * (seq + 1)
    ids = []
    for row in ds:
        ids.extend(enc.encode(row["text"]))
        if len(ids) >= need:
            break
    ids = ids[:need]
    t = torch.tensor(ids, dtype=torch.int32)
    return t[: n_batches * batch * (seq + 1)].view(n_batches, batch, seq + 1)


def count(net):
    return sum(p.numel() for p in net.parameters())


def make_opt(net):
    opt = Terry(lr=0.01, beta=0.9, wd=0.01, target_rms=0.3)
    blocks = getattr(net, "blocks", None)
    if blocks is not None and len(blocks) > 1:
        for i, blk in enumerate(blocks):
            s = 1.0 + 0.3 - 2 * 0.3 * i / (len(blocks) - 1)
            opt.add_param_group(blk.parameters(), scale=s)
        rest = [p for n, p in net.named_parameters() if not n.startswith("blocks.")]
        opt.add_param_group(rest, scale=1.0)
    else:
        opt.add_param_group(net.parameters())
    if hasattr(net, "probes"):
        opt.add_param_group(net.probes.parameters(), scale=1.0)
    return opt


def fair_score(net, x, y):
    with torch.no_grad():
        logits, _ = net(x)
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item()


def markdown_table(out, built):
    lines = ["| arch | final | ms/step | params | flops/tok | kv/tok | note |",
             "|------|------:|--------:|-------:|----------:|-------:|------|"]
    for name, net in built:
        r = out[name]
        lines.append(
            f"| {name} | {r['curve'][-1]:.4f} | {r.get('ms_per_step', 0):.1f} "
            f"| {r['params_m']:.2f}M | {flops_per_token(net) / 1e6:.1f}M "
            f"| {kv_bytes_per_token(net) / 1024:.1f}K | {DESCRIBE.get(name, '')} |")
    return "\n".join(lines)


def vram_math(net):
    n = count(net)
    mb = n * 4 / 1e6
    return round(mb, 1), round(mb, 1)


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--out", type=str, default="Veri/training/benchmark/zoo_1k.json")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--log_every", type=int, default=250)
    ap.add_argument("--only", type=str, default="",
                    help="comma arch names, runs just these")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    out = {}
    if args.resume and Path(args.out).exists():
        out = json.loads(Path(args.out).read_text())
        print(f"resume: {len(out)} archs on file", flush=True)
    want = {n.strip() for n in args.only.split(",")} if args.only else None

    print("streaming tinystories...", flush=True)
    batches = load_batches(args.steps, args.batch, args.seq)
    print(f"batches ready: {tuple(batches.shape)} {batches.element_size() * batches.numel() / 1e6:.0f}MB", flush=True)

    nets, opts, names = [], [], []
    for build in BUILDERS:
        print(f"BUILD {build.__name__}...", flush=True)
        name, net = build()
        if want is not None and name not in want:
            del net
            continue
        net = net.to(device)
        print(f"LIVE {name} {count(net) / 1e6:.2f}M params", flush=True)
        nets.append(net)
        opts.append(make_opt(net))
        names.append(name)
        if name not in out:
            out[name] = {"curve": []}
    Path(args.out).write_text(json.dumps(out, indent=1))

    full_len = args.steps // args.log_every
    training = [len(out.get(n, {}).get("curve", [])) < full_len for n in names]
    if sum(training) < len(names):
        skipped = [n for n, t in zip(names, training) if not t]
        print(f"SKIP {len(skipped)} finished: {', '.join(skipped)}", flush=True)
    start = 0

    t0 = time.time()
    step_s = [0.0] * len(nets)
    for s in range(start, args.steps):
        lr = cosine_schedule(s, args.steps, 0.01, warmup=20)
        b = batches[s].to(device, dtype=torch.long)
        x, y = b[:, :-1], b[:, 1:]
        for i, (net, opt) in enumerate(zip(nets, opts)):
            if not training[i]:
                continue
            opt.lr = lr
            if hasattr(net, "future_scale"):
                net.future_scale = min(1.0, s / 1000)
            opt.zero_grad()
            a = time.perf_counter()
            _, loss = net(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            step_s[i] += time.perf_counter() - a
        if (s + 1) % args.log_every == 0:
            scores = fair_scores(names, nets, x, y)
            for n in names:
                if training[names.index(n)]:
                    out[n]["curve"].append(round(scores[n], 4))
            el = (time.time() - t0) / 60
            best = min(scores, key=scores.get)
            line = " ".join(f"{n}={scores[n]:.2f}" for n in names)
            print(f"step {s + 1:5d}/{args.steps} ({el:.0f}m) best {best} {scores[best]:.4f} | {line}", flush=True)
            Path(args.out).write_text(json.dumps(out, indent=1))

    for name, net in zip(names, nets):
        w_mb, s_mb = vram_math(net)
        i = names.index(name)
        ms = step_s[i] / max(1, args.steps - start) * 1000 if training[i] else out[name].get("ms_per_step", 0)
        out[name].update({"params_m": round(count(net) / 1e6, 2),
                          "ms_per_step": round(ms, 1), "weights_mb": w_mb,
                          "states_mb": s_mb})
        print(f"VRAM {name} {w_mb + s_mb:.0f}MB", flush=True)
    Path(args.out).write_text(json.dumps(out, indent=1))

    if device == "cuda":
        print(f"peak vram: {torch.cuda.max_memory_allocated() / 1e9:.2f}GB", flush=True)
    res = {n: out[n]["curve"][-1] for n in names if out[n].get("curve")}
    win = min(res, key=res.get)
    print(f"FINAL winner: {win} {res[win]:.4f}", flush=True)


def fair_scores(names, nets, x, y):
    out = {}
    with torch.no_grad():
        for name, net in zip(names, nets):
            logits, _ = net(x)
            out[name] = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item()
    return out


if __name__ == "__main__":
    main()
