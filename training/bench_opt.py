import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from minigpt import MiniGPT  # noqa: E402
from terry import Terry, cosine_schedule  # noqa: E402
from muon import Muon  # noqa: E402


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
    t = torch.tensor(ids, dtype=torch.long)
    t = t[: n_batches * batch * (seq + 1)].view(n_batches, batch, seq + 1)
    return t


def opt_bytes(opt, name):
    if hasattr(opt, "state_bytes"):
        return opt.state_bytes()
    if name.startswith("terry"):
        total = 0
        for v in opt.state.values():
            if isinstance(v, dict):
                total += v["q"].numel() * v["q"].element_size()
                total += v["s"].numel() * v["s"].element_size()
            else:
                total += v.numel() * v.element_size()
        for v in getattr(opt, "_err", {}).values():
            total += v.numel() * v.element_size()
        for st in opt._tab.values():
            total += st["r"].numel() * st["r"].element_size()
            if st["c"] is not None:
                total += st["c"].numel() * st["c"].element_size()
        for st in opt._mat.values():
            total += st["r"].numel() * st["r"].element_size()
            if st["c"] is not None:
                total += st["c"].numel() * st["c"].element_size()
        return total
    total = 0
    for st in opt.state.values():
        if isinstance(st, dict):
            total += sum(v.numel() * v.element_size() for v in st.values() if hasattr(v, "numel"))
        else:
            total += st.numel() * st.element_size()
    return total


def train_one(name, peak_lr, batches, steps, device):
    import os

    torch.manual_seed(7)
    if os.environ.get("DUEL_MODEL", "mini") == "m140":
        net = MiniGPT(layers=12, heads=12, dim=768, seq=96).to(device)
    else:
        net = MiniGPT().to(device)
    n_params = net.count()
    if name.startswith("terry"):
        import os

        rms = float(os.environ.get("TERRY_RMS", "0.2"))
        beta = float(os.environ.get("TERRY_BETA", "0.95"))
        depth = float(os.environ.get("TERRY_DEPTH", "0.0"))
        warmup = int(os.environ.get("TERRY_WARMUP", "100"))
        nest_early = os.environ.get("TERRY_NEST_EARLY", "0") == "1"
        escale = float(os.environ.get("TERRY_EMBED_SCALE", "1.0"))
        opt = Terry(lr=peak_lr, beta=beta, wd=0.01, target_rms=rms,
                    warmup_steps=warmup, nesterov_early=nest_early,
                    table_mode=os.environ.get("TERRY_TABLE_MODE", "sign"),
                    table_nesterov=os.environ.get("TERRY_TABLE_NEST", "0") == "1",
                    matrix_mode=os.environ.get("TERRY_MATRIX_MODE", "ns"),
                    table_beta2=float(os.environ.get("TERRY_TABLE_BETA2", "0.99")),
                    table_beta=(float(os.environ["TERRY_TABLE_BETA"])
                                if os.environ.get("TERRY_TABLE_BETA", "") != "" else None))
        if escale != 1.0:
            emb = [p for n, p in net.named_parameters() if "emb" in n]
            rest = [p for n, p in net.named_parameters() if "emb" not in n]
            opt.add_param_group(rest, scale=1.0)
            opt.add_param_group(emb, scale=escale)
        else:
            blocks = getattr(net, "blocks", None)
            if blocks is not None and depth != 0.0 and len(blocks) > 1:
                for i, blk in enumerate(blocks):
                    s = 1.0 - depth + 2 * depth * i / (len(blocks) - 1)
                    opt.add_param_group(blk.parameters(), scale=s)
                rest = [p for n, p in net.named_parameters() if not n.startswith("blocks.")]
                opt.add_param_group(rest, scale=1.0)
            else:
                opt.add_param_group(net.parameters())
    elif name.startswith("muon"):
        import os

        adam_lr = float(os.environ.get("MUON_ADAM_LR", "3e-4"))
        sign = name == "muon_sign"
        nest = os.environ.get("MUON_NESTEROV", "1") == "1"
        opt = Muon(lr=peak_lr, beta=0.95, wd=0.01, adam_lr=adam_lr,
                   nesterov=nest, embed_path="sign" if sign else "adam")
        opt.add_param_group(net.parameters())
    else:
        opt = torch.optim.AdamW(net.parameters(), lr=peak_lr, weight_decay=0.1)
    curve = []
    t0 = time.time()
    la_k = int(os.environ.get("LOOKAHEAD_K", "5"))
    la_a = float(os.environ.get("LOOKAHEAD_A", "0.5"))
    slow = None
    if name == "terry_la":
        slow = [p.detach().clone() for p in net.parameters()]
    for s in range(steps):
        lr = cosine_schedule(s, steps, peak_lr, warmup=20)
        if name.startswith("terry"):
            opt.lr = lr
            opt.lr = lr
        elif name.startswith("muon"):
            import os

            adam_peak = float(os.environ.get("MUON_ADAM_LR", "3e-4"))
            opt.lr = cosine_schedule(s, steps, peak_lr, warmup=20)
            for pg in opt.adam.param_groups:
                pg["lr"] = cosine_schedule(s, steps, adam_peak, warmup=20)
        else:
            for pg in opt.param_groups:
                pg["lr"] = lr
        b = batches[s].to(device)
        x, y = b[:, :-1], b[:, 1:]
        opt.zero_grad()
        _, loss = net(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if slow is not None and (s + 1) % la_k == 0:
            with torch.no_grad():
                for p, q in zip(net.parameters(), slow):
                    q.add_(p.detach() - q, alpha=la_a)
                    p.copy_(q)
        if s == 5:
            print(f"{name:12s} state bytes after step 5: {opt_bytes(opt, name) / 1e6:.1f}MB", flush=True)
        if (s + 1) % 25 == 0 or s == 0:
            curve.append(round(loss.item(), 4))
            print(f"{name:6s} step {s + 1:4d}/{steps} loss {loss.item():.4f} lr {lr:.2e}", flush=True)
    dt = time.time() - t0
    return curve, dt, n_params


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=96)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)
    print("streaming tinystories...", flush=True)
    batches = load_batches(args.steps, args.batch, args.seq)
    print(f"batches ready: {tuple(batches.shape)}", flush=True)

    out = {}
    import os
    OUT = Path(__file__).resolve().parent / "benchmark" / "curves_7m.json"

    muon_lrs = [float(x) for x in os.environ.get("MUON_LRS", "0.02").split(",")]
    terry_lrs = [float(x) for x in os.environ.get("TERRY_LRS", "0.01").split(",")]
    terry_rms = [x.strip() for x in os.environ.get("TERRY_RMSS", os.environ.get("TERRY_RMS", "0.2")).split(",")]
    terry_betas = [x.strip() for x in os.environ.get("TERRY_BETAS", os.environ.get("TERRY_BETA", "0.95")).split(",")]
    terry_escales = [x.strip() for x in os.environ.get("TERRY_ESCALES", os.environ.get("TERRY_EMBED_SCALE", "1.0")).split(",")]
    terry_modes = [x.strip() for x in os.environ.get("TERRY_TABLE_MODES", os.environ.get("TERRY_TABLE_MODE", "sign")).split(",")]
    terry_tbetas = [x.strip() for x in os.environ.get("TERRY_TABLE_BETAS", "").split(",")]
    muon_adam_lrs = [x.strip() for x in os.environ.get("MUON_ADAM_LRS", "3e-4").split(",")]
    adamw_lrs = [float(x) for x in os.environ.get("ADAMW_LRS", "6e-4").split(",")]
    do_ablation = os.environ.get("MUON_ABLATION", "1") == "1"
    _tbs = terry_tbetas if terry_tbetas != [""] else [None]
    grid = (
        [("terry", lr, rms, beta, esc, mode, None, None, None, tb) for lr in terry_lrs for rms in terry_rms
         for beta in terry_betas for esc in terry_escales for mode in terry_modes for tb in _tbs]
        + [("muon", lr, alr, None, None, None, None, None, None, None) for lr in muon_lrs for alr in muon_adam_lrs]
        + [("adamw", lr, None, None, None, None, None, None, None, None) for lr in adamw_lrs]
    )
    if do_ablation:
        grid += [("muon_sign", 0.02, "3e-4", None, None, None, None, None, None, None)]
    if os.environ.get("TERRY_LA", "0") == "1":
        grid += [("terry_la", 0.01, "0.2", "0.9", "1.0", "sign", None, None, None, None)]
    if os.environ.get("TERRY_WAVEF", "0") == "1":
        for tnest, mmat, trust in [("0", "ns", "0.0"), ("1", "ns", "0.0"),
                                   ("0", "graft", "0.0"), ("1", "graft", "0.0"),
                                   ("1", "ns", "2.0"), ("1", "graft", "2.0"),
                                   ("0", "ns", "2.0"), ("0", "graft", "2.0")]:
            grid += [("terry", 0.01, "0.2", "0.9", "1.0", "factored", tnest, mmat, trust, None)]
    if os.environ.get("TERRY_NEWARCH", "0") == "1":
        base = ("terry", 0.02, "0.15", "0.85", "1.0", "factored", "1", "graft", "0.0", None)
        grid += [base]
        os.environ["TERRY_TABLE_BETA"] = ""
        os.environ["TERRY_RESTART"] = "0"
        grid += [("terry-sg", 0.02, "0.15", "0.85", "1.0", "signgraft", "1", "graft", "0.0", None)]
        grid += [("terry-gl", 0.02, "0.15", "0.85", "1.0", "factored", "1", "graft-l", "0.0", None)]
        grid += [("terry-rs", 0.02, "0.15", "0.85", "1.0", "factored", "1", "graft", "0.0", None)]
        grid += [("terry-tb08", 0.02, "0.15", "0.85", "1.0", "factored", "1", "graft", "0.0", None)]
        grid += [("terry-tb099", 0.02, "0.15", "0.99", "1.0", "factored", "1", "graft", "0.0", None)]
    for name, lr, alr, beta, esc, mode, tnest, mmat, trust, tbeta in grid:
        if name.startswith("terry"):
            if alr is not None:
                os.environ["TERRY_RMS"] = alr
            if beta is not None:
                os.environ["TERRY_BETA"] = beta
            if esc is not None:
                os.environ["TERRY_EMBED_SCALE"] = esc
            if mode is not None:
                os.environ["TERRY_TABLE_MODE"] = mode
            os.environ["TERRY_TABLE_NEST"] = tnest if tnest is not None else os.environ.get("TERRY_TABLE_NEST", "0")
            os.environ["TERRY_MATRIX_MODE"] = mmat if mmat is not None else os.environ.get("TERRY_MATRIX_MODE", "ns")
            os.environ["TERRY_TRUST"] = trust if trust is not None else os.environ.get("TERRY_TRUST", "0.0")
            if name == "terry-rs":
                os.environ["TERRY_RESTART"] = "1"
                os.environ["TERRY_TABLE_BETA"] = ""
            elif name == "terry-tb08":
                os.environ["TERRY_RESTART"] = "0"
                os.environ["TERRY_TABLE_BETA"] = "0.8"
            elif name == "terry-tb099":
                os.environ["TERRY_RESTART"] = "0"
                os.environ["TERRY_TABLE_BETA"] = "0.99"
            elif name.startswith("terry"):
                os.environ["TERRY_RESTART"] = "0"
                os.environ["TERRY_TABLE_BETA"] = tbeta if tbeta is not None else ""
        if alr is not None and name.startswith("muon"):
            os.environ["MUON_ADAM_LR"] = alr
        if name.startswith("muon"):
            key = f"{name}_lr{lr}_adam{os.environ.get('MUON_ADAM_LR', '3e-4')}"
        elif name.startswith("terry"):
            key = f"{name}_lr{lr}_rms{os.environ.get('TERRY_RMS', '0.2')}_b{os.environ.get('TERRY_BETA', '0.9')}_e{os.environ.get('TERRY_EMBED_SCALE', '1.0')}_t{os.environ.get('TERRY_TABLE_MODE', 'sign')}_tn{os.environ.get('TERRY_TABLE_NEST', '0')}_mm{os.environ.get('TERRY_MATRIX_MODE', 'ns')}_tr{os.environ.get('TERRY_TRUST', '0.0')}_tb{os.environ.get('TERRY_TABLE_BETA', '')}"
        elif name == "adamw" and len(adamw_lrs) > 1:
            key = f"{name}_{lr}"
        else:
            key = name
        curve, dt, n = train_one(name, lr, batches, args.steps, device)
        out[key] = {"curve": curve, "time_s": round(dt, 1), "params_m": round(n / 1e6, 2)}
        print(f"{key} done in {dt / 60:.1f} min, final loss {curve[-1]}", flush=True)
        OUT.write_text(json.dumps(out, indent=2))

    OUT.write_text(json.dumps(out, indent=2))
    res = {k: out[k]["curve"][-1] for k in out}
    win = min(res, key=res.get)
    print(f"FINAL {res}, winner: {win}")
    muon_keys = [k for k in out if k.startswith("muon") and not k.startswith("muon_sign")]
    best_muon = min(muon_keys, key=lambda k: out[k]["curve"][-1])
    out["muon"] = out[best_muon]
    OUT.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
