import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class VeriConfig:
    def __init__(
        self,
        vocab=65536,
        layers=24,
        dim=2048,
        q_heads=16,
        kv_heads=4,
        mlp_ratio=2.67,
        seq=4096,
        rope_theta=10000.0,
        qk_norm=False,
        sliding=None,
        dropout=0.0,
        diff=False,
        norm="rms",
        rope=True,
    ):
        self.vocab = vocab
        self.layers = layers
        self.dim = dim
        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.mlp_hidden = int(dim * mlp_ratio)
        self.seq = seq
        self.rope_theta = rope_theta
        self.qk_norm = qk_norm
        self.sliding = sliding
        self.dropout = dropout
        self.diff = diff
        self.norm = norm
        self.rope = rope


VERI_1B = VeriConfig(seq=4096, qk_norm=True, sliding=2048)

VERI_REMY = VeriConfig(seq=4096, qk_norm=True, sliding=2048, diff=True)
VERI_REMY.mlp_hidden = 5504

BENCH_BASE = VeriConfig(vocab=50257, layers=6, dim=192, q_heads=6, kv_heads=2, seq=96)
BENCH_QKNORM = VeriConfig(vocab=50257, layers=6, dim=192, q_heads=6, kv_heads=2, seq=96, qk_norm=True)
BENCH_SLIDING = VeriConfig(vocab=50257, layers=6, dim=192, q_heads=6, kv_heads=2, seq=96, sliding=48)
BENCH_THETA = VeriConfig(vocab=50257, layers=6, dim=192, q_heads=6, kv_heads=2, seq=96, rope_theta=500000.0)
BENCH_DEEP = VeriConfig(vocab=50257, layers=9, dim=160, q_heads=5, kv_heads=5, seq=96)
BENCH_COMBO = VeriConfig(vocab=50257, layers=6, dim=192, q_heads=6, kv_heads=2, seq=96, qk_norm=True, sliding=48)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, kind="rms"):
        super().__init__()
        self.eps = eps
        self.kind = kind
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        if self.kind == "ln" or os.environ.get("VERI_NORM", "rms") == "ln":
            return F.layer_norm(x, (x.size(-1),), self.weight, None, self.eps)
        xf = x.float()
        out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (out * self.weight.float()).to(x.dtype)


class FP8StaticLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b):
        out_f = w.shape[0]
        xs = x.shape
        xf = x.reshape(-1, xs[-1])
        if not hasattr(w, "_fp8_s"):
            ax = xf.detach().abs().max().clamp_min(1e-12).float()
            aw = w.detach().float().abs().max().clamp_min(1e-12)
            w._fp8_s = (ax / 448.0, aw / 448.0)
        sx, sw = w._fp8_s
        xq = (xf / sx.to(xf.dtype)).to(torch.float8_e4m3fn)
        wq = (w.t() / sw.to(w.dtype)).to(torch.float8_e4m3fn).contiguous()
        y = torch._scaled_mm(xq, wq, out_dtype=x.dtype, scale_a=sx, scale_b=sw)
        ctx.save_for_backward(xq, wq, sx, sw)
        ctx.xs = xs
        ctx.w_dtype = w.dtype
        ctx.has_b = b is not None
        y = y.view(*xs[:-1], out_f)
        return y + b if b is not None else y

    @staticmethod
    def backward(ctx, go):
        xq, wq, sx, sw = ctx.saved_tensors
        gf = go.reshape(-1, go.shape[-1])
        sgo = gf.detach().abs().max().clamp_min(1e-12).float() / 448.0
        goq = (gf / sgo.to(gf.dtype)).to(torch.float8_e4m3fn)
        dx = torch._scaled_mm(goq, wq.t().contiguous(), out_dtype=go.dtype,
                              scale_a=sgo, scale_b=sw)
        gt = goq.transpose(0, 1).contiguous()
        dw = torch._scaled_mm(gt, xq, out_dtype=torch.float32,
                              scale_a=sgo, scale_b=sx)
        db = gf.sum(0).to(ctx.w_dtype) if ctx.has_b else None
        return dx.view(*ctx.xs), dw.to(ctx.w_dtype), db


def may_fp8(lin, x):
    if os.environ.get("VERI_FP8", "0") == "1":
        return FP8StaticLinear.apply(x, lin.weight, lin.bias)
    return lin(x)


_ROPE_CACHE = {}
_MASK_CACHE = {}


def build_rope(seq, head_dim, theta=10000.0, device="cpu"):
    key = (seq, head_dim, float(theta), str(device))
    hit = _ROPE_CACHE.get(key)
    if hit is not None:
        return hit
    pos = torch.arange(seq, dtype=torch.float, device=device)
    freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float, device=device) / head_dim))
    ang = torch.outer(pos, freq)
    out = (torch.cos(ang), torch.sin(ang))
    _ROPE_CACHE[key] = out
    return out


def apply_rope(x, cos, sin):
    x1, x2 = x.chunk(2, dim=-1)
    c = cos[: x.size(2)].unsqueeze(0).unsqueeze(0)
    s = sin[: x.size(2)].unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)


def repeat_kv(x, n):
    if n == 1:
        return x
    b, h, t, d = x.shape
    x = x.unsqueeze(3).expand(b, h, t, n, d)
    return x.permute(0, 1, 3, 2, 4).reshape(b, h * n, t, d)


def causal_window_mask(t, window=None, device="cpu"):
    key = (t, window, str(device))
    hit = _MASK_CACHE.get(key)
    if hit is not None:
        return hit
    row = torch.arange(t, device=device).unsqueeze(1)
    col = torch.arange(t, device=device).unsqueeze(0)
    ok = col <= row
    if window is not None:
        ok = ok & ((row - col) < window)
    out = torch.where(ok, 0.0, float("-inf"))
    _MASK_CACHE[key] = out
    return out


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.q_heads = cfg.q_heads
        self.kv_heads = cfg.kv_heads
        self.head_dim = cfg.dim // cfg.q_heads
        self.rep = cfg.q_heads // cfg.kv_heads
        assert cfg.dim % cfg.q_heads == 0
        assert cfg.q_heads % cfg.kv_heads == 0
        self.q = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.k = nn.Linear(cfg.dim, cfg.kv_heads * self.head_dim, bias=False)
        self.v = nn.Linear(cfg.dim, cfg.kv_heads * self.head_dim, bias=False)
        self.o = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim, kind=getattr(cfg, "norm", "rms")) if cfg.qk_norm else None
        self.k_norm = RMSNorm(self.head_dim, kind=getattr(cfg, "norm", "rms")) if cfg.qk_norm else None
        if cfg.diff:
            assert cfg.q_heads % 2 == 0
        self.lam = nn.Parameter(torch.zeros(cfg.q_heads // 2) + 0.5) if cfg.diff else None
        self.do_rope = getattr(cfg, "rope", True)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cos, sin, window=None):
        b, t, _ = x.shape
        h, hd = self.q_heads, self.head_dim
        q = may_fp8(self.q, x).view(b, t, h, hd).transpose(1, 2)
        k = may_fp8(self.k, x).view(b, t, self.kv_heads, hd).transpose(1, 2)
        v = may_fp8(self.v, x).view(b, t, self.kv_heads, hd).transpose(1, 2)
        if self.q_norm is not None and os.environ.get("VERI_NOQK", "0") != "1":
            q = self.q_norm(q)
            k = self.k_norm(k)
        if self.do_rope and os.environ.get("VERI_NOROPE", "0") != "1":
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
        k = repeat_kv(k, self.rep)
        v = repeat_kv(v, self.rep)
        mask = causal_window_mask(t, window, x.device).to(x.dtype)
        if self.lam is not None and os.environ.get("VERI_NODIFF", "0") != "1":
            h2 = h // 2
            y1 = F.scaled_dot_product_attention(q[:, :h2], k[:, :h2], v[:, :h2], attn_mask=mask)
            y2 = F.scaled_dot_product_attention(q[:, h2:], k[:, h2:], v[:, h2:], attn_mask=mask)
            lam = self.lam.sigmoid().view(1, h2, 1, 1)
            y = torch.cat([y1 - lam * y2, y2], dim=1)
        else:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        y = y.transpose(1, 2).contiguous().view(b, t, h * hd)
        return self.drop(may_fp8(self.o, y))


class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate = nn.Linear(cfg.dim, cfg.mlp_hidden, bias=False)
        self.up = nn.Linear(cfg.dim, cfg.mlp_hidden, bias=False)
        self.down = nn.Linear(cfg.mlp_hidden, cfg.dim, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(may_fp8(self.down, F.silu(may_fp8(self.gate, x)) * may_fp8(self.up, x)))


class Block(nn.Module):
    def __init__(self, cfg, sliding=None):
        super().__init__()
        self.n1 = RMSNorm(cfg.dim, kind=getattr(cfg, "norm", "rms"))
        self.n2 = RMSNorm(cfg.dim, kind=getattr(cfg, "norm", "rms"))
        self.attn = Attention(cfg)
        self.mlp = SwiGLU(cfg)
        self.sliding = sliding

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin, self.sliding)
        x = x + self.mlp(self.n2(x))
        return x


class VeriTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab, cfg.dim)
        self.pos_drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            [Block(cfg, sliding=cfg.sliding if i % 2 == 0 else None) for i in range(cfg.layers)]
        )
        self.norm = RMSNorm(cfg.dim)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02 / math.sqrt(2 * self.cfg.layers))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None, hidden=False):
        b, t = idx.shape
        cos, sin = build_rope(t, self.cfg.dim // self.cfg.q_heads, self.cfg.rope_theta, idx.device)
        x = self.pos_drop(self.tok_emb(idx))
        cos, sin = cos.to(x.dtype), sin.to(x.dtype)
        ckpt = os.environ.get("VERI_CKPT", "0") == "1"
        for blk in self.blocks:
            x = checkpoint(blk, x, cos, sin, use_reentrant=False) if ckpt else blk(x, cos, sin)
        h = self.norm(x)
        if hidden:
            return h, None
        logits = F.linear(h, self.tok_emb.weight)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def forward_loss_chunked(self, idx, targets, mb=8, tc=2048):
        b, t = idx.shape
        cos, sin = build_rope(t, self.cfg.dim // self.cfg.q_heads, self.cfg.rope_theta, idx.device)
        x = self.pos_drop(self.tok_emb(idx))
        cos, sin = cos.to(x.dtype), sin.to(x.dtype)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        h = self.norm(x)
        W = self.tok_emb.weight
        scale = 1.0 / max(1, b * t)
        total = 0.0
        if W.grad is None:
            W.grad = torch.zeros_like(W)
        for i in range(0, b, mb):
            for j in range(0, t, tc):
                hs = h[i : i + mb, j : j + tc].reshape(-1, h.size(-1))
                tg = targets[i : i + mb, j : j + tc].reshape(-1)
                hs_leaf = hs.detach().requires_grad_(True)
                logits = F.linear(hs_leaf, W).float()
                lt = F.cross_entropy(logits, tg, reduction="sum")
                dh_leaf, dw = torch.autograd.grad(lt * scale, (hs_leaf, W))
                if h.grad is None:
                    h.grad = torch.zeros_like(h)
                sl = h.grad[i : i + mb, j : j + tc]
                sl += dh_leaf.view(sl.shape)
                W.grad += dw
                total += lt.item()
        return total / max(1, b * t), h

    def count(self):
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(self, idx, max_new, temp=1.0, top_k=0):
        self.eval()
        for _ in range(max_new):
            use = idx[:, -self.cfg.seq :]
            logits, _ = self(use)
            last = logits[:, -1, :] / max(temp, 1e-6)
            if top_k > 0:
                if last.max() > last.median() + 1e-4:
                    v, _ = torch.topk(last, min(top_k, last.size(-1)))
                    last[last < v[:, [-1]]] = float("-inf")
            probs = F.softmax(last, dim=-1)
            nxt = torch.multinomial(probs, 1) if temp > 0 else probs.argmax(-1, keepdim=True)
            idx = torch.cat([idx, nxt], dim=1)
        self.train()
        return idx


if __name__ == "__main__":
    for name, cfg in [
        ("base", BENCH_BASE),
        ("qknorm", BENCH_QKNORM),
        ("sliding", BENCH_SLIDING),
        ("theta", BENCH_THETA),
        ("deep", BENCH_DEEP),
        ("combo", BENCH_COMBO),
        ("1b", VERI_1B),
        ("remy", VERI_REMY),
    ]:
        m = VeriTransformer(cfg)
        print(f"{name:8s} {m.count() / 1e6:8.2f}M params")
