import glob
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tokenizer"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "kernels"))

from model import VeriTransformer, VERI_REMY  # noqa: E402
from terry import Terry, cosine_schedule  # noqa: E402
from timmy import Timmy  # noqa: E402

MICRO = 1024
SEQ = 1024
ACCUM = 1
STEPS = int(os.environ.get("VERI_STEPS", "10"))
SAVE_EVERY = int(os.environ.get("VERI_SAVE_EVERY", "9"))
TOK_DIRS = [d for d in os.environ.get("VERI_TOK_DIRS", "./tok").split(",") if d]
TIMMY = os.environ.get("VERI_TIMMY", str(Path(__file__).resolve().parent.parent / "tokenizer" / "timmy.json"))
PEAK_LR = 0.003
WD = 0.01
CHUNK_MB = 64
QUERY_EVERY = 10
QUERY_PROMPTS = ["The cat sat on the", "France is a country",
                 "Scientists have discovered that"]
QUERY_TOKS = 32
QUERY_TEMP = 0.7
QUERY_TOPK = 50
QUERY_FREQ_PEN = 0.2


class CkptBlock(torch.nn.Module):
    def __init__(self, blk):
        super().__init__()
        self.blk = blk

    def forward(self, x, cos, sin):
        from torch.utils.checkpoint import checkpoint

        return checkpoint(self.blk, x, cos, sin, use_reentrant=False)


def main(rank, world):
    t0 = time.time()
    dev = "cuda"

    def log(m):
        if rank == 0:
            print(f"[{time.time() - t0:7.1f}s] [dual] {m}", flush=True)

    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl")
    log(f"nccl world={dist.get_world_size()} rank={rank}")

    timmy = Timmy(TIMMY)
    cfg = VERI_REMY
    cfg.seq = SEQ
    cfg.vocab = timmy.vocab_size()
    net = VeriTransformer(cfg).to(dev, dtype=torch.bfloat16)
    log(f"params {net.count() / 1e9:.2f}b remy seq={SEQ} micro={MICRO} world={world}")

    import model as _model
    from swiglu import swiglu_fused as _sw
    from norm import rmsnorm_fused as _rn
    from norm import rope_fused as _ro

    def _fused_swiglu_forward(self, x):
        return self.drop(self.down(_sw(self.gate(x), self.up(x))))

    _model.SwiGLU.forward = _fused_swiglu_forward
    _model.apply_rope = _ro

    class _FusedRMS(torch.nn.Module):
        def __init__(self, dim, eps=1e-6):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(dim))
            self.eps = eps

        def forward(self, x):
            return _rn(x, self.weight, self.eps)

    n_swap = 0
    for _mod in list(net.modules()):
        if type(_mod).__name__ == "RMSNorm" and hasattr(_mod, "weight"):
            _new = _FusedRMS(_mod.weight.numel(), _mod.eps).to(
                _mod.weight.device, _mod.weight.dtype)
            _found = False
            for _par in net.modules():
                for _nm, _ch in list(_par.named_children()):
                    if _ch is _mod:
                        setattr(_par, _nm, _new)
                        _found = True
                        break
                if _found:
                    break
            n_swap += 1
    log(f"graft: swiglu+rope patched, {n_swap} rmsnorms fused, compile on top")

    net.blocks = torch.nn.ModuleList([CkptBlock(b) for b in net.blocks])
    log("eager, no inductor: cublas shapes rule")

    opt = Terry(lr=PEAK_LR, beta=0.9, wd=WD, target_rms=0.3,
                warmup_steps=100, nesterov_early=True,
                table_mode="factored", table_nesterov=True, matrix_mode="graft")
    from terry import TERRY_VERSION
    log(f"opt {TERRY_VERSION} table={opt.table_mode} mmat={opt.matrix_mode} pen={QUERY_FREQ_PEN}")
    for i, blk in enumerate(list(net.blocks)):
        s = 1.0 + 0.3 - 2 * 0.3 * i / (len(list(net.blocks)) - 1)
        opt.add_param_group(blk.parameters(), scale=s)
    opt.add_param_group([p for n, p in net.named_parameters() if not n.startswith("blocks.")], scale=1.0)

    ARCH = os.environ.get("VERI_ARCH", "a")
    RESUME = os.environ.get("VERI_RESUME", "0") == "1"
    CKPT_DIR = os.environ.get("VERI_CKPT_DIR", f"./ckpt/{ARCH}")
    if rank == 0:
        os.makedirs(CKPT_DIR, exist_ok=True)
    dist.barrier()

    def _cpu(x):
        return x.cpu() if torch.is_tensor(x) else x

    def bundle(step, consumed):
        sd = opt.state_dict()
        sd["moments"] = [{k: _cpu(v) for k, v in m.items()} if isinstance(m, dict) else _cpu(m) for m in sd["moments"]]
        sd["scales"] = [_cpu(s) for s in sd["scales"]]
        return {"net": {k: _cpu(v) for k, v in net.state_dict().items()},
                "opt": sd, "step": step, "consumed": consumed, "arch": ARCH,
                "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state()}

    flat_params = [p for blk in net.blocks for p in blk.parameters()] + \
        [p for n, p in net.named_parameters() if not n.startswith("blocks.")]
    start_step, consumed0 = 1, 0
    if RESUME:
        cands = sorted(glob.glob(f"{CKPT_DIR}/ckpt_*.pt") + glob.glob(f"{CKPT_DIR}/full_*.pt"),
                       key=lambda p: int(p.split("_")[-1].split(".")[0]))
        if not cands:
            log("resume asked, no ckpts, starting fresh")
        else:
            bd = torch.load(cands[-1], map_location="cpu", weights_only=False)
            net.load_state_dict(bd["net"])
            opt.load_state_dict(flat_params, bd["opt"])
            opt.table_mode = "factored"
            opt.matrix_mode = "graft"
            opt.table_nesterov = True
            torch.set_rng_state(bd["rng"])
            torch.cuda.set_rng_state(bd["cuda_rng"])
            start_step, consumed0 = bd["step"] + 1, bd["consumed"]
            log(f"resumed {ARCH} from step {bd['step']}")
    dist.barrier()

    files = []
    for _d in TOK_DIRS:
        files += sorted(glob.glob(_d.rstrip("/") + "/part_*.npy"))
    log(f"shards {len(files)} from {TOK_DIRS}")
    per = (SEQ + 1) * MICRO

    def gen():
        for fp in files:
            toks = torch.from_numpy(np.load(fp)).long()
            n = toks.numel() // per
            for i in range(n):
                yield toks[i * per : (i + 1) * per].view(MICRO, SEQ + 1)

    data = gen()
    for _ in range(consumed0):
        next(data)

    @torch.no_grad()
    def query(step):
        net.eval()
        for pr in QUERY_PROMPTS:
            ids = torch.tensor([timmy.encode(pr)], device=dev)
            for _ in range(QUERY_TOKS):
                logits, _ = net(ids[:, -SEQ:])
                lg = logits[:, -1, :].float()
                if QUERY_FREQ_PEN > 0:
                    counts = torch.bincount(ids[0], minlength=lg.size(-1)).float()
                    lg = lg - QUERY_FREQ_PEN * counts
                lg = lg / QUERY_TEMP
                if QUERY_TOPK > 0:
                    v, _ = torch.topk(lg, QUERY_TOPK)
                    lg = torch.where(lg < v[:, -1:], float("-inf"), lg)
                nxt = torch.softmax(lg, dim=-1).multinomial(1)
                ids = torch.cat([ids, nxt], dim=1)
            log(f"query@{step} [{pr}]: {timmy.decode(ids[0].tolist())}")
        net.train()

    opt.zero_grad()
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
    done_step = STEPS
    for step in range(start_step, STEPS + 1):
        try:
            packs = [next(data) for _ in range(ACCUM * world)]
        except StopIteration:
            log(f"DATA EXHAUSTED after {step - 1} steps, upload more shards. ckpts intact.")
            done_step = step - 1
            break
        mine = packs[rank::world]
        s_t = time.time()
        tot = 0.0
        for b in mine:
            b = b.to(dev, non_blocking=True)
            x, y = b[:, :-1], b[:, 1:]
            ev[0].record()
            tot_f, h = net.forward_loss_chunked(x, y, CHUNK_MB, SEQ)
            ev[1].record()
            torch.autograd.backward(h, h.grad)
            ev[2].record()
            tot += tot_f / ACCUM
        for p in net.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad)
                p.grad /= world
        gnorm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0).item()
        opt.lr = PEAK_LR
        opt.step()
        opt.zero_grad()
        tot_t = torch.tensor(tot, device=dev)
        dist.all_reduce(tot_t)
        tot = (tot_t / world).item()
        if rank == 0 and step % SAVE_EVERY == 0 and step != STEPS:
            torch.save(bundle(step, consumed0 + (step - start_step + 1) * ACCUM * world), f"{CKPT_DIR}/ckpt_{step}.pt")
            log(f"ckpt saved step {step} ({CKPT_DIR})")
        ev[3].record()
        torch.cuda.synchronize()
        f_ms = ev[0].elapsed_time(ev[1])
        b_ms = ev[1].elapsed_time(ev[2])
        o_ms = ev[2].elapsed_time(ev[3])
        dt = time.time() - s_t
        tps = MICRO * SEQ * ACCUM * world / dt
        peak = torch.cuda.max_memory_allocated() / 1e9
        log(f"step {step}/{STEPS} loss {tot:.4f} gnorm {gnorm:.2f} {tps / 1e3:.0f}k tok/s mfu {6 * net.count() * MICRO * SEQ * ACCUM * world / dt / 2.2e15 / world * 100:.1f}% peak {peak:.1f}GB fwd {f_ms:.0f}ms bwd {b_ms:.0f}ms opt {o_ms:.0f}ms")
        if step % QUERY_EVERY == 0:
            if rank == 0:
                query(step)
            dist.barrier()

    if rank == 0:
        torch.save(bundle(done_step, consumed0 + (done_step - start_step + 1) * ACCUM * world), f"{CKPT_DIR}/full_{done_step}.pt")
        log(f"done, full saved ({CKPT_DIR}/full_{done_step}.pt)")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main(int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]))
