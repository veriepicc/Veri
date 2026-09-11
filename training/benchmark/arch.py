import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "model"))

from model import VeriConfig, RMSNorm, build_rope, apply_rope, repeat_kv, SwiGLU, Attention, Block, VeriTransformer
from remy import RemyAttention, RemyBlock, RemyBackbone


# arch zoo, 24 world-class shots at dethroning the transformer
# every entry keeps the llama skeleton (rope, rmsnorm, tied head, terry-ready)
# so winners plug straight into vllm, unsloth kernels, and veri-1b with zero porting
# every builder seeds 7 so identical shapes share identical inits. fair fights only
#
# section map:
#   1. configs, the fixed ~12m budget every entry must respect
#   2. accounting, param match check plus flops and kv bytes per arch
#   3. blocks, the mutations (norms, gates, routing, compression, looping)
#   4. backbones, generic shell plus the two exotic ones (remy-mtp, looped-act)
#   5. builders, one per entry, each returns (name, model)


# ---------------------------------------------------------------- configs ---

# budget math at dim 448: embeddings 50257 x 448 = 22.5m, twelve blocks ~2.1m each
# war-2 runs 50m so flagship clones get real expert sizes and matmuls big
# enough to feed the b300. exact counts print at bench time, tolerance 15pct
ZOO = VeriConfig(vocab=50257, layers=12, dim=448, q_heads=8, kv_heads=2, seq=128)
ZOO_QK = VeriConfig(vocab=50257, layers=12, dim=448, q_heads=8, kv_heads=2, seq=128, qk_norm=True)

# grind config: the 98k-tok/s B300 recipe. plain attention (no diff),
# no qk norms, no rope, layernorm. speed demon, brain unverified.
# zoo decides if it learns. same dims as ZOO_QK, counts match.
ZOO_GRIND = VeriConfig(vocab=50257, layers=12, dim=448, q_heads=8, kv_heads=2,
                       seq=128, qk_norm=False, diff=False, norm="ln", rope=False)
ZOO_COMBO = VeriConfig(vocab=50257, layers=12, dim=448, q_heads=8, kv_heads=2, seq=128, qk_norm=True, sliding=64)

# wide-shallow: 8 fat layers. narrow-deep: 20 thin layers. both ~50m
WIDE = VeriConfig(vocab=50257, layers=8, dim=528, q_heads=8, kv_heads=2, seq=128, qk_norm=True)
NARROW = VeriConfig(vocab=50257, layers=20, dim=352, q_heads=8, kv_heads=2, seq=128, qk_norm=True)
# kv sweeps: mqa shares one kv head, mha8 gives every q head its own
MQA = VeriConfig(vocab=50257, layers=12, dim=448, q_heads=8, kv_heads=1, seq=128, qk_norm=True)
MHA8 = VeriConfig(vocab=50257, layers=12, dim=448, q_heads=8, kv_heads=8, seq=128, qk_norm=True)
# looped box: single shared block, dim picked so one block plus embeddings ~46m
LOOP = VeriConfig(vocab=50257, layers=1, dim=768, q_heads=12, kv_heads=3, seq=128, qk_norm=True, mlp_ratio=3.0)
# flagship budgets: dim 416 keeps moe entries near 50m with real expert width
DS41 = VeriConfig(vocab=50257, layers=12, dim=384, q_heads=8, kv_heads=2,
                 seq=128, rope_theta=10000.0, qk_norm=True)
KIMI = VeriConfig(vocab=50257, layers=12, dim=384, q_heads=8, kv_heads=2,
                  seq=128, rope_theta=10000.0, qk_norm=True)
OSS120_CFG = VeriConfig(vocab=50257, layers=12, dim=416, q_heads=8, kv_heads=2,
                    seq=128, rope_theta=150000.0, qk_norm=False, sliding=64)
GLM53_CFG = VeriConfig(vocab=50257, layers=12, dim=416, q_heads=8, kv_heads=2,
                   seq=128, rope_theta=8000000.0, qk_norm=True)
GEMMA4 = VeriConfig(vocab=50257, layers=12, dim=448, q_heads=8, kv_heads=2,
                    seq=128, rope_theta=10000.0, qk_norm=False)
QWEN35 = VeriConfig(vocab=50257, layers=12, dim=432, q_heads=8, kv_heads=2,
                    seq=128, rope_theta=10000000.0, qk_norm=True)

BUDGET_M = 50.0
BUDGET_TOL = 0.15


# ------------------------------------------------------------- accounting ---

def count_params(net):
    # every param, no exceptions, embeddings included since they dominate here
    return sum(p.numel() for p in net.parameters())


def state_bytes_est(net):
    # terry fp32 one-state estimate, what the bench actually allocates
    return sum(p.numel() * 4 for p in net.parameters())


def flops_per_token(net):
    # rough fwd flops per token. attn linears 8*d^2 per layer (q,k,v,o at d),
    # mlp 6*d*h, plus causal score math 2*t*d per layer. good to 2x, fine
    cfg = net.cfg
    t = cfg.seq
    d = cfg.dim
    h = cfg.mlp_hidden
    per_layer = 8 * d * d + 6 * d * h + 2 * t * d
    n_layers = len(getattr(net, "blocks", [1]))
    # looped unrolls one block kmax times, count the unroll not the params
    kmax = getattr(net, "kmax", None)
    if kmax is not None:
        per_layer = per_layer * kmax
    return n_layers * per_layer


def kv_bytes_per_token(net):
    # exact kv cache per token in bf16. walks attention modules so mla-lite
    # reports its latent footprint instead of the gqa formula. looped keeps
    # one block of kv slots reused frame by frame, so count a single block
    per_layer = []
    for m in net.modules():
        if isinstance(m, MLAAttention):
            per_layer.append(m.cache_per_token)
        elif isinstance(m, Attention) or isinstance(m, RemyAttention) or isinstance(m, GatedAttention):
            cfg = net.cfg
            per_layer.append(cfg.kv_heads * (cfg.dim // cfg.q_heads) * 2 * 2)
    if not per_layer:
        cfg = net.cfg
        per_layer = [cfg.kv_heads * (cfg.dim // cfg.q_heads) * 2 * 2]
    if getattr(net, "kmax", None) is not None:
        # looped reuses one frame of slots no matter the unroll depth
        return per_layer[0]
    return sum(per_layer)


def report_table(built):
    # ascii scorecard, params plus flops plus kv. paste into benchmark.md
    print(f"{'arch':12s} {'params':>8s} {'flops/tok':>10s} {'kv/tok':>8s} {'match':>6s}")
    for name, net in built:
        n = count_params(net)
        ok = abs(n / 1e6 - BUDGET_M) / BUDGET_M <= BUDGET_TOL
        print(f"{name:12s} {n / 1e6:7.2f}M {flops_per_token(net) / 1e6:9.1f}M "
              f"{kv_bytes_per_token(net) / 1024:7.1f}K {'ok' if ok else 'OFF':>6s}")


# ----------------------------------------------------------------- blocks ---

class ActMLP(nn.Module):
    # same three matrices as swiglu, different gate curve. param count identical
    # gelu is the gpt-2 classic, relu is sparser and sometimes wins small
    def __init__(self, cfg, act="gelu"):
        super().__init__()
        self.gate = nn.Linear(cfg.dim, cfg.mlp_hidden, bias=False)
        self.up = nn.Linear(cfg.dim, cfg.mlp_hidden, bias=False)
        self.down = nn.Linear(cfg.mlp_hidden, cfg.dim, bias=False)
        tanh_gelu = lambda z: 0.5 * z * (1 + torch.tanh(
            math.sqrt(2 / math.pi) * (z + 0.044715 * z * z * z)))
        self.act = {"gelu": F.gelu, "relu": F.relu, "silu": F.silu,
                    "tanh_gelu": tanh_gelu}[act]

    def forward(self, x):
        return self.down(self.act(self.gate(x)) * self.up(x))


class LSBlock(nn.Module):
    # layerscale: learned per-channel gate on each branch, init tiny
    # branches start near zero so signal flows clean, gates open as needed
    # deit proved this lets deep stacks train without warmup drama
    def __init__(self, cfg, attn_cls=Attention, init=1e-2):
        super().__init__()
        self.n1 = RMSNorm(cfg.dim)
        self.n2 = RMSNorm(cfg.dim)
        self.attn = attn_cls(cfg)
        self.mlp = SwiGLU(cfg)
        self.g1 = nn.Parameter(torch.ones(cfg.dim) * init)
        self.g2 = nn.Parameter(torch.ones(cfg.dim) * init)

    def forward(self, x, cos, sin):
        x = x + self.g1 * self.attn(self.n1(x), cos, sin)
        x = x + self.g2 * self.mlp(self.n2(x))
        return x


class PostBlock(nn.Module):
    # normformer: extra norm after attn and after mlp before the residual add
    # residual stream stays unit scale no matter what the branches do
    # costs two norms per layer, crumbs, stability goes way up
    def __init__(self, cfg, attn_cls=Attention):
        super().__init__()
        self.n1 = RMSNorm(cfg.dim)
        self.n2 = RMSNorm(cfg.dim)
        self.p1 = RMSNorm(cfg.dim)
        self.p2 = RMSNorm(cfg.dim)
        self.attn = attn_cls(cfg)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.p1(self.attn(self.n1(x), cos, sin))
        x = x + self.p2(self.mlp(self.n2(x)))
        return x


class SandwichBlock(nn.Module):
    # sandwich norm: norm on both sides of every branch, pre and post
    # strictest signal control in the zoo. if qknorm is a seatbelt this is a cage
    def __init__(self, cfg, attn_cls=Attention):
        super().__init__()
        self.n1 = RMSNorm(cfg.dim)
        self.n2 = RMSNorm(cfg.dim)
        self.p1 = RMSNorm(cfg.dim)
        self.p2 = RMSNorm(cfg.dim)
        self.attn = attn_cls(cfg)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.p1(self.attn(self.n1(x), cos, sin))
        h = self.n2(x)
        x = x + self.p2(self.mlp(h))
        # second pre-norm on the mlp path input, hence sandwich
        return x


class ParallelBlock(nn.Module):
    # palm style: attn and mlp read the same normed input in one stream
    # one norm instead of two, depth halves, width effectively doubles
    # communication-heavy clusters love it, one less sync point per layer
    def __init__(self, cfg, attn_cls=Attention):
        super().__init__()
        self.n = RMSNorm(cfg.dim)
        self.attn = attn_cls(cfg)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        h = self.n(x)
        return x + self.attn(h, cos, sin) + self.mlp(h)


class ActBlock(nn.Module):
    # activation swap lab. backbone identical to base so the act takes full blame
    def __init__(self, cfg, act="gelu"):
        super().__init__()
        self.n1 = RMSNorm(cfg.dim)
        self.n2 = RMSNorm(cfg.dim)
        self.attn = Attention(cfg)
        self.mlp = ActMLP(cfg, act)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin)
        x = x + self.mlp(self.n2(x))
        return x


class MLAAttention(nn.Module):
    # mla-lite, deepseek flavor without the training bill
    # kv squeezed to latent 16 per head, normed there, up-projected at use
    # cache per token halves versus gqa-2. same attention math after the lift
    # latent 16 is aggressive on purpose, the bench decides if it survives
    def __init__(self, cfg, latent=16):
        super().__init__()
        self.q_heads = cfg.q_heads
        self.kv_heads = cfg.kv_heads
        self.head_dim = cfg.dim // cfg.q_heads
        self.rep = cfg.q_heads // cfg.kv_heads
        assert cfg.q_heads % cfg.kv_heads == 0
        self.lat = latent
        self.q = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.k_down = nn.Linear(cfg.dim, cfg.kv_heads * latent, bias=False)
        self.k_up = nn.Linear(cfg.kv_heads * latent, cfg.kv_heads * self.head_dim, bias=False)
        self.v_down = nn.Linear(cfg.dim, cfg.kv_heads * latent, bias=False)
        self.v_up = nn.Linear(cfg.kv_heads * latent, cfg.kv_heads * self.head_dim, bias=False)
        self.k_norm = RMSNorm(latent)
        self.v_norm = RMSNorm(latent)
        self.q_norm = RMSNorm(self.head_dim) if cfg.qk_norm else None
        self.o = nn.Linear(cfg.dim, cfg.dim, bias=False)
        # exact cache footprint per token for this layer, bf16 kv
        self.cache_per_token = cfg.kv_heads * latent * 2 * 2

    def forward(self, x, cos, sin, window=None):
        b, t, _ = x.shape
        h, hd = self.q_heads, self.head_dim
        q = self.q(x).view(b, t, h, hd).transpose(1, 2)
        # squeeze, scrub, lift. the norm on latent is what makes mla trainable
        kl = self.k_norm(self.k_down(x).view(b, t, self.kv_heads, self.lat))
        vl = self.v_norm(self.v_down(x).view(b, t, self.kv_heads, self.lat))
        k = self.k_up(kl.flatten(2)).view(b, t, self.kv_heads, hd).transpose(1, 2)
        v = self.v_up(vl.flatten(2)).view(b, t, self.kv_heads, hd).transpose(1, 2)
        if self.q_norm is not None:
            q = self.q_norm(q)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        k = repeat_kv(k, self.rep)
        v = repeat_kv(v, self.rep)
        from model import causal_window_mask
        mask = causal_window_mask(t, window, x.device).to(x.dtype)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.o(y.transpose(1, 2).contiguous().view(b, t, h * hd))


class MLABlock(nn.Module):
    def __init__(self, cfg, sliding=None):
        super().__init__()
        self.n1 = RMSNorm(cfg.dim)
        self.n2 = RMSNorm(cfg.dim)
        self.attn = MLAAttention(cfg)
        self.mlp = SwiGLU(cfg)
        self.sliding = sliding

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin, self.sliding)
        x = x + self.mlp(self.n2(x))
        return x


class GatedAttention(Attention):
    # per-token output gate on the o projection
    # heads learn to shut themselves off on tokens where they add noise
    # remy subtracts noise after the fact, this refuses to emit it. cousins
    def __init__(self, cfg):
        super().__init__(cfg)
        self.gate = nn.Linear(cfg.dim, cfg.dim, bias=False)

    def forward(self, x, cos, sin, window=None):
        return torch.sigmoid(self.gate(x)) * super().forward(x, cos, sin, window)


class GatedBlock(nn.Module):
    def __init__(self, cfg, sliding=None):
        super().__init__()
        self.n1 = RMSNorm(cfg.dim)
        self.n2 = RMSNorm(cfg.dim)
        self.attn = GatedAttention(cfg)
        self.mlp = SwiGLU(cfg)
        self.sliding = sliding

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin, self.sliding)
        x = x + self.mlp(self.n2(x))
        return x


class RemyParallelBlock(nn.Module):
    # champ attention meets palm wiring. denoise plus half the depth
    # if remy wins on signal and parallel wins on systems, this wins on both
    def __init__(self, cfg, sliding=None):
        super().__init__()
        self.n = RMSNorm(cfg.dim)
        self.attn = RemyAttention(cfg)
        self.mlp = SwiGLU(cfg)
        self.sliding = sliding

    def forward(self, x, cos, sin):
        h = self.n(x)
        # sliding applies to the shared stream, same interleaving as the backbone
        return x + self.attn(h, cos, sin, self.sliding) + self.mlp(h)


class LDWrapper(nn.Module):
    # layerdrop: skip the whole block with prob p while training
    # forces every layer to be useful alone, no co-adaptation crutches
    # at inference all layers run, free ensemble effect. fan et al 2019
    def __init__(self, blk, p=0.1):
        super().__init__()
        self.blk = blk
        self.p = p

    def parameters(self, *a, **k):
        return self.blk.parameters(*a, **k)

    def named_parameters(self, *a, **k):
        return self.blk.named_parameters(*a, **k)

    def forward(self, x, cos, sin):
        if self.training and torch.rand(()) < self.p:
            return x
        return self.blk(x, cos, sin)


# --------------------------------------------------------------- backbones ---

class ZooTransformer(nn.Module):
    # generic shell: any block class, sliding interleaved on evens like veri
    # kw passes block options (attn_cls, act). tied head, same init as veri
    def __init__(self, cfg, block_cls, drop_p=0.0, **kw):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab, cfg.dim)
        has_sliding = "sliding" in block_cls.__init__.__code__.co_varnames
        blocks = []
        for i in range(cfg.layers):
            if has_sliding:
                blk = block_cls(cfg, sliding=cfg.sliding if i % 2 == 0 else None, **kw)
            else:
                blk = block_cls(cfg, **kw)
            if drop_p > 0:
                blk = LDWrapper(blk, drop_p)
            blocks.append(blk)
        self.blocks = nn.ModuleList(blocks)
        self.norm = RMSNorm(cfg.dim)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02 / math.sqrt(2 * max(1, self.cfg.layers)))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        b, t = idx.shape
        cos, sin = build_rope(t, self.cfg.dim // self.cfg.q_heads, self.cfg.rope_theta, idx.device)
        x = self.tok_emb(idx)
        cos, sin = cos.to(x.dtype), sin.to(x.dtype)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        h = self.norm(x)
        logits = F.linear(h, self.tok_emb.weight)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def count(self):
        return sum(p.numel() for p in self.parameters())


class RemyMTP(RemyBackbone):
    # champ backbone plus multi-token probes. extra futures densify the signal
    # future_scale ramps in the loop so early steps learn next-token first
    def __init__(self, cfg, mtp=4):
        super().__init__(cfg)
        self.mtp = mtp
        self.future_scale = 1.0
        self.probes = nn.ModuleList([nn.Linear(cfg.dim, cfg.dim, bias=False) for _ in range(mtp - 1)])
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02 / math.sqrt(2 * max(1, self.cfg.layers)))
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        b, t = idx.shape
        cos, sin = build_rope(t, self.cfg.dim // self.cfg.q_heads, self.cfg.rope_theta, idx.device)
        x = self.tok_emb(idx)
        cos, sin = cos.to(x.dtype), sin.to(x.dtype)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        h = self.norm(x)
        W = self.tok_emb.weight
        logits = F.linear(h, W)
        loss = None
        if targets is not None:
            if self.mtp <= 1 or targets.size(1) < self.mtp:
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                return logits, loss
            total = F.cross_entropy(
                logits[:, :-self.mtp + 1].reshape(-1, logits.size(-1)),
                targets[:, :-self.mtp + 1].reshape(-1), reduction="mean")
            w = 0.5 * self.future_scale
            for i, probe in enumerate(self.probes, start=1):
                lf = F.linear(probe(h), W)
                k = self.mtp - 1 - i
                if k > 0:
                    a, g = lf[:, :-k], targets[:, i:targets.size(1) - k]
                else:
                    a, g = lf, targets[:, i:]
                n = min(a.size(1), g.size(1))
                total = total + w * F.cross_entropy(
                    a[:, :n].reshape(-1, a.size(-1)), g[:, :n].reshape(-1), reduction="mean")
                w *= 0.5
            loss = total
        return logits, loss


# group 7: legacy flavor builds, superseded by the faithful group 8 below
# kept as free ablations (flavor versus faithful). the outside giants note stands
# (mla compression plus fine-grained moe with one shared expert and aux-loss
# free routing, load balanced by bias instead of a loss term). gpt-oss-flavor
# follows openai's open release (moe every layer, sliding window everywhere,
# high-theta rope with yarn roots, swiglu, no shared expert).
# honest deviations at 12m: 8 experts instead of 128/256 (experts must stay
# big enough to learn anything), conv and mrope skipped, embeddings tied.
# moe counts every expert toward the budget, active per token is far less,
# which flatters moe on paper. the bench scores loss anyway, no hiding
class FineMoE(nn.Module):
    # shared expert handles the common stuff, routed experts take specialties
    # router picks top-k per token. balance via per-expert bias nudged against
    # overload each step, no auxiliary loss to fight the main objective
    def __init__(self, cfg, n_exp=8, top_k=2, exp_ratio=0.25, shared=True):
        super().__init__()
        self.n_exp = n_exp
        self.top_k = top_k
        hidden = max(16, int(cfg.mlp_hidden * exp_ratio))
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(cfg.dim, hidden, bias=False),
                          nn.SiLU(),
                          nn.Linear(hidden, cfg.dim, bias=False))
            for _ in range(n_exp)])
        self.shared = SwiGLU(cfg) if shared else None
        self.router = nn.Linear(cfg.dim, n_exp, bias=False)
        self.register_buffer("bias", torch.zeros(n_exp))
        self.bias_lr = 1e-3

    def forward(self, x):
        flat = x.reshape(-1, x.size(-1))
        logits = self.router(flat) + self.bias
        vals, idx = logits.topk(self.top_k, dim=-1)
        weights = torch.softmax(vals, dim=-1)
        out = self.shared(x) if self.shared is not None else 0.0
        exp_out = torch.zeros_like(flat)
        for e, expert in enumerate(self.experts):
            hit = (idx == e)
            if bool(hit.any()):
                w = (weights * hit).sum(-1, keepdim=True)
                exp_out = exp_out + w * expert(flat)
        out = out + exp_out.view_as(x)
        # aux-free balance: overloaded experts get their bias trimmed
        if self.training:
            with torch.no_grad():
                loads = torch.zeros(self.n_exp, device=x.device)
                for e in range(self.n_exp):
                    loads[e] = (idx == e).float().mean()
                target = torch.full_like(loads, self.top_k / self.n_exp)
                self.bias.add_(torch.sign(target - loads) * self.bias_lr)
        return out


class DSBlock(nn.Module):
    # deepseek-flavor layer: mla squeeze plus fine-grained moe
    def __init__(self, cfg, sliding=None):
        super().__init__()
        self.n1 = RMSNorm(cfg.dim)
        self.n2 = RMSNorm(cfg.dim)
        self.attn = MLAAttention(cfg)
        self.mlp = FineMoE(cfg, n_exp=8, top_k=2, exp_ratio=0.25, shared=True)
        self.sliding = sliding

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin, self.sliding)
        x = x + self.mlp(self.n2(x))
        return x


OSS = VeriConfig(vocab=50257, layers=6, dim=192, q_heads=6, kv_heads=2,
                 seq=96, rope_theta=150000.0, qk_norm=True, sliding=48)


class OSSBlock(nn.Module):
    # gpt-oss-flavor layer: sliding gqa everywhere, moe instead of dense mlp
    # no shared expert, router alone decides. window 48 at seq 96 is the
    # scaled-down shadow of their 128 window at 128k context
    def __init__(self, cfg, sliding=None):
        super().__init__()
        self.n1 = RMSNorm(cfg.dim)
        self.n2 = RMSNorm(cfg.dim)
        self.attn = Attention(cfg)
        self.mlp = FineMoE(cfg, n_exp=8, top_k=2, exp_ratio=0.33, shared=False)
        self.sliding = sliding if sliding is not None else 48

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin, self.sliding)
        x = x + self.mlp(self.n2(x))
        return x


class LoopedACT(nn.Module):
    # the 5k layers idea, done honest. one shared block looped up to kmax times
    # with per-step film adapters so every iteration behaves differently
    # plus act halting: each token stops thinking when its confident (graves 2016)
    # easy tokens exit at step 3, hard tokens burn all 16. ponder cost keeps it real
    # params stay flat no matter the depth. depth becomes compute, not memory
    def __init__(self, cfg, kmax=16, tau=1e-3, halt_eps=1e-2):
        super().__init__()
        self.cfg = cfg
        self.kmax = kmax
        self.tau = tau
        self.halt_eps = halt_eps
        self.tok_emb = nn.Embedding(cfg.vocab, cfg.dim)
        self.block = Block(cfg)
        # film per step, zero-init so step 0 starts as the plain block
        self.gamma = nn.Parameter(torch.zeros(kmax, cfg.dim))
        self.beta = nn.Parameter(torch.zeros(kmax, cfg.dim))
        self.halt = nn.Linear(cfg.dim, 1, bias=True)
        nn.init.constant_(self.halt.bias, -2.0)
        self.norm = RMSNorm(cfg.dim)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear) and m is not self.halt:
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        b, t = idx.shape
        cos, sin = build_rope(t, self.cfg.dim // self.cfg.q_heads, self.cfg.rope_theta, idx.device)
        x = self.tok_emb(idx)
        cos, sin = cos.to(x.dtype), sin.to(x.dtype)
        # act loop. still flag, cumulated halt mass, ponder counts live tokens
        cum = x.new_zeros(b, t)
        out = x.new_zeros(b, t, x.size(-1))
        ponder = x.new_zeros(b, t)
        for k in range(self.kmax):
            x = x * (1.0 + self.gamma[k]) + self.beta[k]
            h = self.block(x, cos, sin)
            p = torch.sigmoid(self.halt(h)).squeeze(-1)
            live = (cum < 1.0 - self.halt_eps).to(x.dtype)
            w = p * live
            out = out + w.unsqueeze(-1) * h
            cum = cum + w
            ponder = ponder + live
            x = h
            if bool((cum >= 1.0 - self.halt_eps).all()):
                break
        # remainder mass lands on the final state, standard act bookkeeping
        rem = (1.0 - cum).clamp_min(0.0)
        out = out + rem.unsqueeze(-1) * x
        ponder = ponder + rem
        h = self.norm(out)
        logits = F.linear(h, self.tok_emb.weight)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            loss = loss + self.tau * ponder.mean()
        return logits, loss

    def count(self):
        return sum(p.numel() for p in self.parameters())


# ------------------------------------------- group 9: frontier lite-cuts ---
# faithful cuts of 2026 frontier models, mechanisms preserved, counts
# downscaled, every deviation documented. verified against the orgs'
# cards + configs + modeling code before porting. tinystories, seed 7,
# same terry as everyone else. the champ defends on this table.

class Step5BlockS(nn.Module):
    # sliding layer: sliding-64 flex attention + clamped moe.
    # clamp 7.0 exact (their swiglu_limits), scaling 3.0 exact, shared
    # expert exact, router bias exact (aux-free, in TopKMoE already).
    def __init__(self, cfg):
        super().__init__()
        self.n1 = RMSNorm(cfg.dim)
        self.n2 = RMSNorm(cfg.dim)
        self.attn = FlexAttention(cfg)
        self.mlp = TopKMoE(cfg, n_exp=8, top_k=2, exp_hidden=64,
                            scaling=3.0, shared=True, swiglu_limit=7.0)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin, window=64)
        x = x + self.mlp(self.n2(x))
        return x


class Step5BlockF(nn.Module):
    # full layer: partial rope (third of dims, exact fraction) + csa-style
    # sparse mask from the indexer (their sparse_config on full layers) +
    # clamped moe. theta kept 10k (real full layers run 10m+yarn, documented:
    # yarn is identity at seq 128 anyway, theta gap is the honest deviation).
    def __init__(self, cfg, indexer):
        super().__init__()
        self.n1 = RMSNorm(cfg.dim)
        self.n2 = RMSNorm(cfg.dim)
        self.attn = FlexAttention(cfg)
        self.mlp = TopKMoE(cfg, n_exp=8, top_k=2, exp_hidden=64,
                            scaling=3.0, shared=True, swiglu_limit=7.0)
        self.indexer = indexer

    def forward(self, x, cos, sin, mask=None):
        b, t, _ = x.shape
        h = self.n1(x)
        q = self.attn.q(h).view(b, t, self.attn.q_heads, self.attn.head_dim).transpose(1, 2)
        k = self.attn.k(h).view(b, t, self.attn.kv_heads, self.attn.head_dim).transpose(1, 2)
        v = self.attn.v(h).view(b, t, self.attn.kv_heads, self.attn.head_dim).transpose(1, 2)
        # partial tables sliced to the rotated width (shared tables are full-dim)
        rd = (q.shape[-1] // 3) // 2 * 2
        cosf, sinf = cos[:, :rd // 2].contiguous(), sin[:, :rd // 2].contiguous()
        q = partial_rope_apply(q, cosf, sinf, 1.0 / 3.0)
        k = partial_rope_apply(k, cosf, sinf, 1.0 / 3.0)
        k = repeat_kv(k, self.attn.rep)
        v = repeat_kv(v, self.attn.rep)
        from model import causal_window_mask
        m = causal_window_mask(t, None, x.device).to(x.dtype)
        if mask is not None:
            m = m + mask.to(x.dtype)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
        y = self.attn.o(y.transpose(1, 2).contiguous().view(b, t, -1))
        x = x + y
        x = x + self.mlp(self.n2(x))
        return x


class Step5Backbone(nn.Module):
    # 12 layers, 3 sliding + 1 full repeating (real: same 3:1 rhythm over 92).
    # indexer feeds every full layer (their sparse on fulls). 1 mtp probe
    # (real: 3 predict layers, documented). tied head.
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab, cfg.dim)
        self.indexer = GLMIndexer(cfg)
        self.blocks = nn.ModuleList()
        for i in range(cfg.layers):
            if i % 4 == 3:
                self.blocks.append(Step5BlockF(cfg, self.indexer))
            else:
                self.blocks.append(Step5BlockS(cfg))
        self.norm = RMSNorm(cfg.dim)
        self.probe = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.future_scale = 1.0
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02 / math.sqrt(2 * max(1, self.cfg.layers)))
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        b, t = idx.shape
        cos, sin = build_rope(t, self.cfg.dim // self.cfg.q_heads, self.cfg.rope_theta, idx.device)
        x = self.tok_emb(idx)
        cos, sin = cos.to(x.dtype), sin.to(x.dtype)
        from model import causal_window_mask
        base = causal_window_mask(t, None, x.device).to(x.dtype)
        for blk in self.blocks:
            if isinstance(blk, Step5BlockF):
                topk = self.indexer(x, cos, sin)
                sel = torch.zeros(b, t, t, dtype=torch.bool, device=x.device)
                sel.scatter_(-1, topk.clamp_max(t - 1), True)
                mask = (base + torch.where(sel, 0.0, torch.tensor(float("-inf"), device=x.device))).unsqueeze(1)
                x = blk(x, cos, sin, mask=mask)
            else:
                x = blk(x, cos, sin)
        h = self.norm(x)
        W = self.tok_emb.weight
        logits = F.linear(h, W)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            if t > 1:
                lf = F.linear(self.probe(h[:, :-1]), W)
                n = min(lf.size(1), targets.size(1) - 1)
                loss = loss + 0.3 * self.future_scale * F.cross_entropy(
                    lf[:, :n].reshape(-1, lf.size(-1)),
                    targets[:, 1:n + 1].reshape(-1), reduction="mean")
        return logits, loss

    def count(self):
        return sum(p.numel() for p in self.parameters())


class InklingBackbone(nn.Module):
    # 12 layers, sliding-64 sliding-64 full repeating (real interleave rhythm
    # undisclosed, documented guess). 8 experts top-2 sigmoid + 1 shared
    # (real: 256 top-6 + 2 shared, documented downscale). tied head.
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab, cfg.dim)
        self.blocks = nn.ModuleList()
        for i in range(cfg.layers):
            win = None if i % 3 == 2 else 64
            self.blocks.append(GatedBlock(cfg, sliding=win))
        for blk in self.blocks:
            blk.mlp = TopKMoE(cfg, n_exp=8, top_k=2, exp_hidden=64, shared=True)
        self.norm = RMSNorm(cfg.dim)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02 / math.sqrt(2 * max(1, self.cfg.layers)))
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        b, t = idx.shape
        cos, sin = build_rope(t, self.cfg.dim // self.cfg.q_heads, self.cfg.rope_theta, idx.device)
        x = self.tok_emb(idx)
        cos, sin = cos.to(x.dtype), sin.to(x.dtype)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        h = self.norm(x)
        logits = F.linear(h, self.tok_emb.weight)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def count(self):
        return sum(p.numel() for p in self.parameters())


class MambaLite(nn.Module):
    # mamba-2 selective scan, eager loop. expand 2 and conv kernel 4 exact
    # (their config). d_state 16 not 128 and dim-level not head-level
    # (documented downscales for an eager bench). slow but honest.
    def __init__(self, dim, d_state=16, conv_k=4):
        super().__init__()
        self.d_inner = 2 * dim
        self.d_state = d_state
        self.in_proj = nn.Linear(dim, 2 * self.d_inner, bias=False)
        self.conv = nn.Conv1d(self.d_inner, self.d_inner, conv_k, groups=self.d_inner, bias=False, padding=conv_k - 1)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        self.A_log = nn.Parameter(torch.randn(self.d_inner, d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)

    def forward(self, x):
        b, t, _ = x.shape
        xz = self.in_proj(x)
        xx, zz = xz.chunk(2, dim=-1)
        xx = self.conv(xx.transpose(1, 2))[:, :, :t].transpose(1, 2)
        xx = F.silu(xx)
        dt = F.softplus(self.dt_proj(xx)) + 1e-3
        A = -torch.exp(self.A_log)
        state = xx.new_zeros(b, self.d_inner, self.d_state)
        ys = []
        for i in range(t):
            dti = dt[:, i, :].unsqueeze(-1)
            xxi = xx[:, i, :].unsqueeze(-1)
            state = state * torch.exp(dti * A.unsqueeze(0)) + dti * xxi
            ys.append((state.sum(-1) + self.D.unsqueeze(0) * xx[:, i, :]).unsqueeze(1))
        y = torch.cat(ys, dim=1)
        return self.out_proj(y * F.silu(zz))


class NemoBlockM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n1 = RMSNorm(cfg.dim)
        self.n2 = RMSNorm(cfg.dim)
        self.mix = MambaLite(cfg.dim)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.mix(self.n1(x))
        x = x + self.mlp(self.n2(x))
        return x


class NemoBlockE(nn.Module):
    # moe-only layer (their block map alternates mamba/moe/attn as layer
    # TYPES, moe layers carry no attention). scaling 2.5 exact, norm topk
    # exact, 1 shared exact.
    def __init__(self, cfg):
        super().__init__()
        self.n = RMSNorm(cfg.dim)
        self.mlp = TopKMoE(cfg, n_exp=8, top_k=2, exp_hidden=64,
                            scaling=2.5, shared=True)

    def forward(self, x, cos, sin):
        return x + self.mlp(self.n(x))


class NemoBlockA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n1 = RMSNorm(cfg.dim)
        self.n2 = RMSNorm(cfg.dim)
        self.attn = FlexAttention(cfg)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin)
        x = x + self.mlp(self.n2(x))
        return x


class NemotronBackbone(nn.Module):
    # 12 layers: attention every 6th (their 52-map rhythm, verified from
    # config), moe-only between, mamba elsewhere. theta 10k exact. 1 mtp
    # probe (theirs ships mtp layers). tied head (theirs untied, documented:
    # untied eats half the bench budget).
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab, cfg.dim)
        self.blocks = nn.ModuleList()
        for i in range(cfg.layers):
            if i % 6 == 5:
                self.blocks.append(NemoBlockA(cfg))
            elif i % 2 == 1:
                self.blocks.append(NemoBlockE(cfg))
            else:
                self.blocks.append(NemoBlockM(cfg))
        self.norm = RMSNorm(cfg.dim)
        self.probe = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.future_scale = 1.0
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02 / math.sqrt(2 * max(1, self.cfg.layers)))
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        b, t = idx.shape
        cos, sin = build_rope(t, self.cfg.dim // self.cfg.q_heads, self.cfg.rope_theta, idx.device)
        x = self.tok_emb(idx)
        cos, sin = cos.to(x.dtype), sin.to(x.dtype)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        h = self.norm(x)
        W = self.tok_emb.weight
        logits = F.linear(h, W)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            if t > 1:
                lf = F.linear(self.probe(h[:, :-1]), W)
                n = min(lf.size(1), targets.size(1) - 1)
                loss = loss + 0.3 * self.future_scale * F.cross_entropy(
                    lf[:, :n].reshape(-1, lf.size(-1)),
                    targets[:, 1:n + 1].reshape(-1), reduction="mean")
        return logits, loss

    def count(self):
        return sum(p.numel() for p in self.parameters())


class ConvMixer(nn.Module):
    # double-gated short conv, lite. depthwise kernel 3 exact (their cache 3).
    # single sigmoid gate here, their second gate folded in (documented).
    def __init__(self, dim, kernel=3):
        super().__init__()
        self.dw = nn.Conv1d(dim, dim, kernel, groups=dim, bias=False, padding=kernel - 1)
        self.gate = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        b, t, d = x.shape
        h = self.dw(x.transpose(1, 2))[:, :, :t].transpose(1, 2)
        return h * torch.sigmoid(self.gate(x))


class LiquidBlockC(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n1 = RMSNorm(cfg.dim)
        self.n2 = RMSNorm(cfg.dim)
        self.mix = ConvMixer(cfg.dim)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.mix(self.n1(x))
        x = x + self.mlp(self.n2(x))
        return x


class LiquidBlockA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n1 = RMSNorm(cfg.dim)
        self.n2 = RMSNorm(cfg.dim)
        self.attn = FlexAttention(cfg)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin)
        x = x + self.mlp(self.n2(x))
        return x


class LiquidBackbone(nn.Module):
    # 12 layers, conv conv attention repeating (their 30-layer map rhythm,
    # verified from config: 22 conv + 8 full). full attention, no sliding
    # (verified). theta 10m exact, tied embeddings exact. the conv-dominant
    # bet: can short convs carry three quarters of the stack.
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab, cfg.dim)
        self.blocks = nn.ModuleList(
            [LiquidBlockA(cfg) if i % 3 == 2 else LiquidBlockC(cfg) for i in range(cfg.layers)])
        self.norm = RMSNorm(cfg.dim)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02 / math.sqrt(2 * max(1, self.cfg.layers)))
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        b, t = idx.shape
        cos, sin = build_rope(t, self.cfg.dim // self.cfg.q_heads, self.cfg.rope_theta, idx.device)
        x = self.tok_emb(idx)
        cos, sin = cos.to(x.dtype), sin.to(x.dtype)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        h = self.norm(x)
        logits = F.linear(h, self.tok_emb.weight)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def count(self):
        return sum(p.numel() for p in self.parameters())


STEP5 = VeriConfig(vocab=50257, layers=12, dim=400, q_heads=8, kv_heads=2, seq=128, qk_norm=False)
INKLING = VeriConfig(vocab=50257, layers=12, dim=400, q_heads=8, kv_heads=2, seq=128, qk_norm=False)
NEMO = VeriConfig(vocab=50257, layers=12, dim=400, q_heads=8, kv_heads=2, seq=128, qk_norm=False)
LIQUID = VeriConfig(vocab=50257, layers=12, dim=448, q_heads=8, kv_heads=2, seq=128,
                    rope_theta=10000000.0, qk_norm=False)
GRANITE20 = VeriConfig(vocab=50257, layers=20, dim=352, q_heads=8, kv_heads=2, seq=128,
                       rope_theta=10000000.0, qk_norm=False)


# ---------------------------------------------------------------- builders ---

def _seed():
    torch.manual_seed(7)


# group 1: references. base is the llama skeleton, qknorm the 200-step champ,
# combo adds sliding, remy is the 1k-step champ every entry must beat
def b_base(): _seed(); return ("base", VeriTransformer(ZOO))
def b_qknorm(): _seed(); return ("qknorm", VeriTransformer(ZOO_QK))
def b_combo(): _seed(); return ("combo", VeriTransformer(ZOO_COMBO))
def b_remy(): _seed(); return ("remy", RemyBackbone(ZOO_QK))
def b_remy_grind(): _seed(); return ("remy-grind", VeriTransformer(ZOO_GRIND))

# group 2: stability stackers. layerscale gates branches, post-norm scrubs the
# stream, sandwich does both. each crossed with plain and remy attention
def b_remy_ls(): _seed(); return ("remy-ls", ZooTransformer(ZOO_QK, LSBlock, attn_cls=RemyAttention))
def b_remy_post(): _seed(); return ("remy-post", ZooTransformer(ZOO_QK, PostBlock, attn_cls=RemyAttention))
def b_remy_sand(): _seed(); return ("remy-sand", ZooTransformer(ZOO_QK, SandwichBlock, attn_cls=RemyAttention))
def b_qk_post(): _seed(); return ("qk-post", ZooTransformer(ZOO_QK, PostBlock, attn_cls=Attention))
def b_qk_ls(): _seed(); return ("qk-ls", ZooTransformer(ZOO_QK, LSBlock, attn_cls=Attention))

# group 3: wiring. parallel halves the depth, remy-parallel is champ meets palm
def b_parallel(): _seed(); return ("parallel", ZooTransformer(ZOO, ParallelBlock, attn_cls=Attention))
def b_parallel_qk(): _seed(); return ("parallel-qk", ZooTransformer(ZOO_QK, ParallelBlock, attn_cls=Attention))
def b_remy_parallel(): _seed(); return ("remy-parallel", ZooTransformer(ZOO_QK, RemyParallelBlock))

# group 4: shape. activations swapped at identical counts, kv heads swept,
# wide-shallow and narrow-deep test the depth versus width religion
def b_geglu(): _seed(); return ("geglu", ZooTransformer(ZOO_QK, ActBlock, act="gelu"))
def b_reglu(): _seed(); return ("reglu", ZooTransformer(ZOO_QK, ActBlock, act="relu"))
def b_mqa(): _seed(); return ("mqa", VeriTransformer(MQA))
def b_mha(): _seed(); return ("mha8", VeriTransformer(MHA8))
def b_wide(): _seed(); return ("wide-shallow", VeriTransformer(WIDE))
def b_narrow(): _seed(); return ("narrow-deep", VeriTransformer(NARROW))

# group 5: memory and signal. mla-lite shrinks the cache, gated refuses noise,
# mtp densifies gradients, remy-mtp stacks champ plus signal, layerdrop
# regularizes co-adaptation, looped-act trades params for thinking time
def b_mla(): _seed(); return ("mla-lite", ZooTransformer(ZOO_QK, MLABlock))
def b_gated(): _seed(); return ("gated", ZooTransformer(ZOO_QK, GatedBlock))
def b_remy_mtp(): _seed(); return ("remy-mtp", RemyMTP(ZOO_QK))
def b_layerdrop(): _seed(); return ("layerdrop", ZooTransformer(ZOO_QK, Block, drop_p=0.1))
def b_looped(): _seed(); return ("looped-act", LoopedACT(LOOP, kmax=8))
def b_qwen35(): _seed(); return ("qwen35", Qwen35(QWEN35))
def b_deepseek(): _seed(); return ("deepseek-v3", ZooTransformer(ZOO_QK, DSBlock))
def b_gptoss(): _seed(); return ("gpt-oss-flavor", ZooTransformer(OSS, OSSBlock))
def b_dsv41(): _seed(); return ("ds-v41", DSv41(DS41))
def b_oss120(): _seed(); return ("oss-120", OSS120(OSS120_CFG))
def b_kimi(): _seed(); return ("kimi-k3", KimiBackbone(KIMI))
def b_glm(): _seed(); return ("glm-53", GLM53(GLM53_CFG))
def b_gemma(): _seed(); return ("gemma-4", Gemma4(GEMMA4))

# group 9: frontier lite-cuts. verified against cards/configs/code, see classes
def b_step5(): _seed(); return ("step-5", Step5Backbone(STEP5))
def b_inkling(): _seed(); return ("inkling", InklingBackbone(INKLING))
def b_nemotron(): _seed(); return ("nemotron-mamba", NemotronBackbone(NEMO))
def b_liquid(): _seed(); return ("liquid-conv", LiquidBackbone(LIQUID))
def b_granite(): _seed(); return ("granite-40", ZooTransformer(GRANITE20, Block))


# group 6: the outside baseline. qwen3.5 as published: 3 gated-deltanet
# linear layers per full gated attention, output gates everywhere, no bias,
# rope theta 10m, swiglu, one mtp head. two honest deviations at this budget:
# embeddings stay tied (untied costs 9.6m, the whole budget) and the short
# conv kernel on k/v is skipped (512 params of fiddliness, essence is delta)
class GatedDeltaNet(nn.Module):
    # linear attention with a memory that corrects instead of accumulates
    # state S holds kv associations. each step: forget a little (gate),
    # then delta-write the surprise (v minus what S already predicts for k)
    # read with q. O(seq) time, O(1) memory per token. the 1m context cheat
    def __init__(self, cfg):
        super().__init__()
        self.q_heads = cfg.q_heads
        self.kv_heads = cfg.kv_heads
        self.head_dim = cfg.dim // cfg.q_heads
        self.rep = cfg.q_heads // cfg.kv_heads
        assert cfg.q_heads % cfg.kv_heads == 0
        d, hd = cfg.dim, self.head_dim
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, cfg.kv_heads * hd, bias=False)
        self.v = nn.Linear(d, cfg.kv_heads * hd, bias=False)
        self.gate = nn.Linear(d, cfg.q_heads, bias=False)
        self.beta = nn.Linear(d, cfg.q_heads, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.ogate = nn.Linear(d, d, bias=False)
        self.q_norm = RMSNorm(hd) if cfg.qk_norm else None

    def forward(self, x, cos, sin, window=None):
        # rope on q only here, positions ride the queries while memory is raw
        # partial rotary would be the full qwen recipe, full rope is the lite cut
        b, t, _ = x.shape
        h, hd, kv = self.q_heads, self.head_dim, self.kv_heads
        q = self.q(x).view(b, t, h, hd).transpose(1, 2)
        k = self.k(x).view(b, t, kv, hd).transpose(1, 2)
        v = self.v(x).view(b, t, kv, hd).transpose(1, 2)
        if self.q_norm is not None:
            q = self.q_norm(q)
        q = apply_rope(q, cos, sin)
        # key norm is the deltanet signature, unbounded keys blow up the state
        k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
        g = torch.sigmoid(self.gate(x)).view(b, t, h, 1).transpose(1, 2)
        beta = torch.sigmoid(self.beta(x)).view(b, t, h, 1).transpose(1, 2)
        # expand kv groups so every q head owns its memory matrix
        k = repeat_kv(k, self.rep)
        v = repeat_kv(v, self.rep)
        state = q.new_zeros(b, h, hd, hd)
        outs = []
        for i in range(t):
            qi, ki, vi = q[:, :, i], k[:, :, i], v[:, :, i]
            gi, bi = g[:, :, i], beta[:, :, i]
            # delta write: predict, surprise, correct. gate forgets stale junk
            pred = (state * ki.unsqueeze(-1)).sum(-2)
            state = state * gi.unsqueeze(-1) + bi.unsqueeze(-1) * (vi - pred).unsqueeze(-1) * ki.unsqueeze(-2)
            outs.append((state * qi.unsqueeze(-1)).sum(-2))
        y = torch.stack(outs, dim=2).transpose(1, 2).contiguous().view(b, t, h * hd)
        y = self.o(y)
        return torch.sigmoid(self.ogate(x)) * y


class QwenBlock(nn.Module):
    # full gated attention layer, every 4th. gqa plus qk norm plus output gate
    def __init__(self, cfg, sliding=None):
        super().__init__()
        self.n1 = RMSNorm(cfg.dim)
        self.n2 = RMSNorm(cfg.dim)
        self.attn = GatedAttention(cfg)
        self.mlp = SwiGLU(cfg)
        self.sliding = sliding

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin, self.sliding)
        x = x + self.mlp(self.n2(x))
        return x


class DeltaBlock(nn.Module):
    # linear layer, three of four. same norms and mlp, memory instead of scores
    def __init__(self, cfg, sliding=None):
        super().__init__()
        self.n1 = RMSNorm(cfg.dim)
        self.n2 = RMSNorm(cfg.dim)
        self.attn = GatedDeltaNet(cfg)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin)
        x = x + self.mlp(self.n2(x))
        return x


class Qwen35(nn.Module):
    # 8 layers, full gated attention every 4th, deltanet everywhere else
    # one mtp probe like the real thing, ramped so early steps stay clean
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.future_scale = 1.0
        self.tok_emb = nn.Embedding(cfg.vocab, cfg.dim)
        blocks = []
        for i in range(cfg.layers):
            if (i + 1) % 4 == 0:
                blocks.append(QwenBlock(cfg))
            else:
                blocks.append(DeltaBlock(cfg))
        self.blocks = nn.ModuleList(blocks)
        self.norm = RMSNorm(cfg.dim)
        self.probe = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02 / math.sqrt(2 * max(1, self.cfg.layers)))
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        b, t = idx.shape
        cos, sin = build_rope(t, self.cfg.dim // self.cfg.q_heads, self.cfg.rope_theta, idx.device)
        x = self.tok_emb(idx)
        cos, sin = cos.to(x.dtype), sin.to(x.dtype)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        h = self.norm(x)
        W = self.tok_emb.weight
        logits = F.linear(h, W)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            if t > 1:
                lf = F.linear(self.probe(h[:, :-1]), W)
                n = min(lf.size(1), targets.size(1) - 1)
                loss = loss + 0.3 * self.future_scale * F.cross_entropy(
                    lf[:, :n].reshape(-1, lf.size(-1)),
                    targets[:, 1:n + 1].reshape(-1), reduction="mean")
        return logits, loss

    def count(self):
        return sum(p.numel() for p in self.parameters())


# group 8: faithful flagship cuts, built from configs plus modeling code
# rule: mechanisms preserved, counts downscaled, every deviation documented
# ds-v41 from DeepSeek-V4.1-Flash config plus v4 paper. oss-120 from the 120b
# config plus modeling_gpt_oss.py. kimi-k3 from KimiLinearConfig values plus
# modeling_kimi_linear.py. glm-53 from its config plus modeling_glm_moe_dsa.py
# gemma-4 from its text config plus public gemma lineage code

class SiTU(nn.Module):
    # kimi exact port. beta * tanh(gate / beta) * sigmoid(gate), up optionally
    # squashed by linear_beta. k3 runs beta 4.0 and linear 25.0, gentle curves
    def __init__(self, beta=4.0, linear_beta=25.0):
        super().__init__()
        self.beta = beta
        self.linear_beta = linear_beta

    def forward(self, x):
        d = x.shape[-1] // 2
        gate = x[..., :d].float()
        up = x[..., d:].float()
        a = self.beta * torch.tanh(gate / self.beta) * torch.sigmoid(gate)
        if self.linear_beta is not None:
            up = self.linear_beta * torch.tanh(up / self.linear_beta)
        return (a * up).to(x.dtype)


class SituMLP(nn.Module):
    # kimi mlp: gate and up projected, concatenated through situ, down
    def __init__(self, cfg, hidden=None, beta=4.0, linear_beta=25.0):
        super().__init__()
        h = hidden or cfg.mlp_hidden
        self.gate = nn.Linear(cfg.dim, h, bias=False)
        self.up = nn.Linear(cfg.dim, h, bias=False)
        self.down = nn.Linear(h, cfg.dim, bias=False)
        self.act = SiTU(beta, linear_beta)

    def forward(self, x):
        return self.down(self.act(torch.cat([self.gate(x), self.up(x)], dim=-1)))


def yarn_tables(seq, head_dim, theta, factor, orig_ctx, beta_fast=32.0, beta_slow=1.0, device="cpu"):
    # exact yarn ramp from the oss and glm configs. low dims stay linear,
    # high dims interpolate. at seq 128 this is near identity, kept for honesty
    dim = head_dim
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float, device=device) / dim))
    low = dim * math.log(orig_ctx / (beta_slow * 2 * math.pi)) / math.log(theta)
    high = dim * math.log(orig_ctx / (beta_fast * 2 * math.pi)) / math.log(theta)
    ramp = ((torch.arange(dim // 2, dtype=torch.float, device=device) - low) / max(high - low, 1e-6)).clamp_(0, 1)
    inv = 1.0 / factor
    corr = (1 - ramp) * inv + ramp
    freqs = freqs / corr
    pos = torch.arange(seq, dtype=torch.float, device=device)
    ang = torch.outer(pos, freqs)
    return torch.cos(ang), torch.sin(ang)


def partial_rope_apply(x, cos, sin, frac=0.25):
    # v4 style: rope rides the last frac of dims, the rest stays put
    # x is batch heads seq head_dim, cos and sin are built for the rope dims
    r = (x.shape[-1] * frac) // 2 * 2
    r = int(max(r, 2))
    x1, x2 = x[..., :-r], x[..., -r:]
    a, b = x2.chunk(2, dim=-1)
    t = x.size(2)
    c = cos[:t].unsqueeze(0).unsqueeze(0)
    s = sin[:t].unsqueeze(0).unsqueeze(0)
    return torch.cat([x1, torch.cat([a * c - b * s, a * s + b * c], dim=-1)], dim=-1)


class FlexAttention(nn.Module):
    # one gqa body serving three flagships. bias linears for oss, qk norms
    # for ds and gemma-full, learned sink logit per head for oss and ds,
    # optional output gate for ds gated attention. sparse additive mask
    # carries glm topk and ds retrieval without touching this code
    def __init__(self, cfg, bias=False, out_gate=False, use_sink=False, eps=1e-6):
        super().__init__()
        self.q_heads = cfg.q_heads
        self.kv_heads = cfg.kv_heads
        self.head_dim = cfg.dim // cfg.q_heads
        self.rep = cfg.q_heads // cfg.kv_heads
        assert cfg.dim % cfg.q_heads == 0 and cfg.q_heads % cfg.kv_heads == 0
        d, hd = cfg.dim, self.head_dim
        self.q = nn.Linear(d, d, bias=bias)
        self.k = nn.Linear(d, cfg.kv_heads * hd, bias=bias)
        self.v = nn.Linear(d, cfg.kv_heads * hd, bias=bias)
        self.o = nn.Linear(d, d, bias=bias)
        self.q_norm = RMSNorm(hd, eps) if cfg.qk_norm else None
        self.k_norm = RMSNorm(hd, eps) if cfg.qk_norm else None
        self.sinks = nn.Parameter(torch.zeros(cfg.q_heads)) if use_sink else None
        self.ogate = nn.Linear(d, d, bias=bias) if out_gate else None

    def forward(self, x, cos, sin, window=None, sparse_mask=None, rope_frac=1.0):
        b, t, _ = x.shape
        h, hd = self.q_heads, self.head_dim
        q = self.q(x).view(b, t, h, hd).transpose(1, 2)
        k = self.k(x).view(b, t, self.kv_heads, hd).transpose(1, 2)
        v = self.v(x).view(b, t, self.kv_heads, hd).transpose(1, 2)
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        if rope_frac >= 1.0:
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
        else:
            q = partial_rope_apply(q, cos, sin, rope_frac)
            k = partial_rope_apply(k, cos, sin, rope_frac)
        k = repeat_kv(k, self.rep)
        v = repeat_kv(v, self.rep)
        from model import causal_window_mask
        mask = causal_window_mask(t, window, x.device).to(x.dtype)
        if sparse_mask is not None:
            mask = mask + sparse_mask.to(x.dtype)
        if self.sinks is not None:
            s = self.sinks.view(1, h, 1, 1).expand(b, h, t, 1).to(x.dtype)
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(hd) + mask.unsqueeze(0).unsqueeze(0)
            scores = torch.cat([scores, s], dim=-1)
            scores = scores - scores.amax(-1, keepdim=True)
            probs = torch.softmax(scores.float(), dim=-1).to(x.dtype)[..., :-1]
            y = probs @ v
        else:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        y = y.transpose(1, 2).contiguous().view(b, t, h * hd)
        y = self.o(y)
        if self.ogate is not None:
            y = torch.sigmoid(self.ogate(x)) * y
        return y


class TopKMoE(nn.Module):
    # one router chassis for ds, oss, and glm. scoring picks the family:
    # sqrtsoftplus for ds-v41, sigmoid for glm, raw-logit topk plus softmax
    # for oss. norm_topk and scaling match each config. aux-free bias does
    # the balancing everywhere (oss config wants 0.9 aux loss, which would
    # nuke a 12m toy, documented deviation). frozen_router gives hash-moe
    # spirit: static selection, learned scores
    def __init__(self, cfg, n_exp=8, top_k=2, exp_hidden=None, scoring="sigmoid",
                 scaling=1.0, norm_topk=True, shared=True, swiglu_limit=None,
                 oss_style=False, frozen_router=False, bias_terms=False):
        super().__init__()
        self.n_exp = n_exp
        self.top_k = top_k
        self.scoring = scoring
        self.scaling = scaling
        self.norm_topk = norm_topk
        self.oss_style = oss_style
        h = exp_hidden or cfg.mlp_hidden
        self.gate_up = nn.Parameter(torch.empty(n_exp, cfg.dim, 2 * h))
        self.gate_up_bias = nn.Parameter(torch.zeros(n_exp, 2 * h)) if bias_terms else None
        self.down = nn.Parameter(torch.empty(n_exp, h, cfg.dim))
        self.down_bias = nn.Parameter(torch.zeros(n_exp, cfg.dim)) if bias_terms else None
        self.shared = SwiGLU(cfg) if shared else None
        self.router = nn.Linear(cfg.dim, n_exp, bias=bias_terms)
        if frozen_router:
            self.router.weight.requires_grad_(False)
            if self.router.bias is not None:
                self.router.bias.requires_grad_(False)
        self.register_buffer("bias", torch.zeros(n_exp))
        self.limit = swiglu_limit
        self.bias_lr = 1e-3

    def _expert_fn(self, gu, e):
        # interleaved even-odd split, exact oss layout. swiglu path clamps
        # only when a limit is set (ds-v41 limit 10, oss limit 7)
        g, u = gu[..., ::2], gu[..., 1::2]
        if self.oss_style:
            g = g.clamp(max=7.0)
            u = u.clamp(min=-7.0, max=7.0)
            return (u + 1) * (g * torch.sigmoid(1.702 * g))
        if self.limit is not None:
            g = g.clamp(max=self.limit)
            u = u.clamp(min=-self.limit, max=self.limit)
        return F.silu(g) * u

    def forward(self, x):
        flat = x.reshape(-1, x.size(-1))
        logits = self.router(flat.float()).to(x.dtype) + self.bias.to(x.dtype)
        if self.scoring == "sqrtsoftplus":
            scores = torch.sqrt(F.softplus(logits))
        elif self.scoring == "sigmoid":
            scores = torch.sigmoid(logits)
        elif self.scoring == "oss-topk":
            # oss recipe: topk on raw logits, softmax over the winners only
            vals, idx = logits.topk(self.top_k, dim=-1)
            w = torch.softmax(vals, dim=-1)
            out = self.shared(x) if self.shared is not None else 0.0
            acc = torch.zeros_like(flat)
            for e in range(self.n_exp):
                hit = (idx == e)
                if bool(hit.any()):
                    w8 = (w * hit).sum(-1, keepdim=True)
                    gu = flat @ self.gate_up[e]
                    if self.gate_up_bias is not None:
                        gu = gu + self.gate_up_bias[e]
                    acc = acc + w8 * (self._expert_fn(gu, e) @ self.down[e] +
                                      (self.down_bias[e] if self.down_bias is not None else 0))
            return out + acc.view_as(x)
        else:
            scores = torch.softmax(logits, dim=-1)
        vals, idx = (scores + self.bias.to(x.dtype)).topk(self.top_k, dim=-1)
        if self.scoring == "oss-softmax":
            w = torch.softmax(vals, dim=-1)
        else:
            w = scores.gather(1, idx)
            if self.norm_topk and self.top_k > 1:
                w = w / (w.sum(-1, keepdim=True) + 1e-20)
            w = w * self.scaling
        out = self.shared(x) if self.shared is not None else 0.0
        acc = torch.zeros_like(flat)
        for e in range(self.n_exp):
            hit = (idx == e)
            if bool(hit.any()):
                w8 = (w * hit).sum(-1, keepdim=True)
                gu = flat @ self.gate_up[e]
                if self.gate_up_bias is not None:
                    gu = gu + self.gate_up_bias[e]
                acc = acc + w8 * (self._expert_fn(gu, e) @ self.down[e] +
                                  (self.down_bias[e] if self.down_bias is not None else 0))
        out = out + acc.view_as(x)
        # aux-free balance runs even with a frozen router so scores stay honest
        if self.training:
            with torch.no_grad():
                loads = torch.zeros(self.n_exp, device=x.device)
                for e in range(self.n_exp):
                    loads[e] = (idx == e).float().mean()
                target = torch.full_like(loads, self.top_k / self.n_exp)
                self.bias.add_(torch.sign(target - loads) * self.bias_lr)
        return out


class LatentMoE(nn.Module):
    # kimi exact port, downscaled. shared down-proj into latent space, tiny
    # situ experts there, optional norm, shared up-proj, plus shared experts
    # this is how 896 experts fit a budget: the projections are shared, only
    # the crumbs multiply. 64 experts top-8 at latent 48 costs one dense mlp
    def __init__(self, cfg, n_exp=64, top_k=8, latent=64, exp_hidden=64,
                 n_shared=2, shared_hidden=256, scaling=1.0, beta=4.0, linear_beta=25.0):
        super().__init__()
        self.n_exp = n_exp
        self.top_k = top_k
        self.scaling = scaling
        self.down = nn.Linear(cfg.dim, latent, bias=False)
        self.up = nn.Linear(latent, cfg.dim, bias=False)
        self.norm = RMSNorm(latent)
        self.experts = nn.ModuleList([
            KimiExpert(latent, exp_hidden, beta, linear_beta) for _ in range(n_exp)])
        self.shared = nn.ModuleList([SituMLP(cfg, hidden=shared_hidden, beta=beta, linear_beta=linear_beta)
                                     for _ in range(n_shared)])
        self.router = nn.Linear(cfg.dim, n_exp, bias=False)
        self.register_buffer("bias", torch.zeros(n_exp))
        self.bias_lr = 1e-3

    def forward(self, x):
        flat = x.reshape(-1, x.size(-1))
        scores = torch.sigmoid(self.router(flat.float()).to(x.dtype) + self.bias.to(x.dtype))
        vals, idx = (scores + self.bias.to(x.dtype)).topk(self.top_k, dim=-1)
        w = vals / (vals.sum(-1, keepdim=True) + 1e-20) * self.scaling
        h = self.down(flat)
        acc = torch.zeros_like(h)
        for e, expert in enumerate(self.experts):
            hit = (idx == e)
            if bool(hit.any()):
                w8 = (w * hit).sum(-1, keepdim=True)
                acc = acc + w8 * expert(h)
        acc = self.up(self.norm(acc))
        shared_out = sum(s(x) for s in self.shared)
        out = acc.view_as(x) + shared_out
        if self.training:
            with torch.no_grad():
                loads = torch.zeros(self.n_exp, device=x.device)
                for e in range(self.n_exp):
                    loads[e] = (idx == e).float().mean()
                target = torch.full_like(loads, self.top_k / self.n_exp)
                self.bias.add_(torch.sign(target - loads) * self.bias_lr)
        return out


class KimiExpert(nn.Module):
    # one latent expert: gate and up in latent space, situ between, down out
    def __init__(self, latent, hidden, beta=4.0, linear_beta=25.0):
        super().__init__()
        self.gate = nn.Linear(latent, hidden, bias=False)
        self.up = nn.Linear(latent, hidden, bias=False)
        self.down = nn.Linear(hidden, latent, bias=False)
        self.act = SiTU(beta, linear_beta)

    def forward(self, h):
        return self.down(self.act(torch.cat([self.gate(h), self.up(h)], dim=-1)))


def apply_attn_res(prefix_sum, block_residual, proj_w, norm_w, eps=1e-5):
    # kimi exact port. every stored block plus the live prefix votes through
    # a learned softmax, output is the consensus read. cross-layer memory
    # without carrying the whole past, one matmul per read
    v = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
    vf = v.float()
    k = vf * torch.rsqrt(vf.pow(2).mean(-1, keepdim=True) + eps)
    scores = (k * (norm_w.float() * proj_w.float().squeeze(0))).sum(-1)
    probs = scores.softmax(-1).unsqueeze(1)
    return (probs @ vf).squeeze(1).to(v.dtype)


class CSALite(nn.Module):
    # deepseek-v4 compressed sparse attention, lite cut. every query reads
    # exact local sliding window plus top-8 retrieved blocks out of 16.
    # keys mean-pooled per 8-token block, scored by the query, gathered exact
    # real v4 compresses entries with learned weights and shares kv mqa-wide.
    # lite keeps full entries and per-head retrieval, same select-then-attend
    # shape. partial rope on the last quarter of dims, q/kv norms, sink logit
    def __init__(self, cfg, block=8, topk=8, window=48):
        super().__init__()
        self.block = block
        self.topk = topk
        self.window = window
        d, hd = cfg.dim, cfg.dim // cfg.q_heads
        self.q_heads = cfg.q_heads
        self.kv_heads = cfg.kv_heads
        self.rep = cfg.q_heads // cfg.kv_heads
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, cfg.kv_heads * hd, bias=False)
        self.v = nn.Linear(d, cfg.kv_heads * hd, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.q_norm = RMSNorm(hd)
        self.k_norm = RMSNorm(hd)
        self.sinks = nn.Parameter(torch.zeros(cfg.q_heads))
        self.ogate = nn.Linear(d, d, bias=False)

    def forward(self, x, cos, sin):
        b, t, _ = x.shape
        h, hd = self.q_heads, self.q.weight.shape[0] // self.q_heads
        q = self.q(x).view(b, t, h, hd).transpose(1, 2)
        k = self.k(x).view(b, t, self.kv_heads, hd).transpose(1, 2)
        v = self.v(x).view(b, t, self.kv_heads, hd).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = partial_rope_apply(q, cos, sin, 0.25)
        k = partial_rope_apply(k, cos, sin, 0.25)
        k = repeat_kv(k, self.rep)
        v = repeat_kv(v, self.rep)
        from model import causal_window_mask
        # local branch: exact sliding window plus sink, oss-style sink math
        ml = causal_window_mask(t, self.window, x.device).to(x.dtype)
        sc = (q.float() @ k.transpose(-2, -1).float()) / math.sqrt(hd) + ml.unsqueeze(0).unsqueeze(0)
        s = self.sinks.view(1, h, 1, 1).expand(b, h, t, 1).to(x.dtype)
        sc = torch.cat([sc, s], dim=-1)
        sc = sc - sc.amax(-1, keepdim=True)
        y_local = torch.softmax(sc.float(), dim=-1).to(x.dtype)[..., :-1] @ v
        # retrieval branch: pooled block keys, topk per query, exact gather
        nb = t // self.block
        kb = k[:, :, :nb * self.block, :].view(b, h, nb, self.block, hd).mean(3)
        rel = torch.arange(t, device=x.device) // self.block
        s2 = (q.float() @ kb.transpose(-2, -1).float()) / math.sqrt(hd)
        s2 = s2.masked_fill(
            torch.arange(nb, device=x.device).view(1, 1, 1, nb) > rel.view(1, 1, t, 1),
            float("-inf"))
        top = s2.topk(min(self.topk, nb), dim=-1).indices
        tok = (top.unsqueeze(-1) * self.block + torch.arange(self.block, device=x.device))
        tok = tok.flatten(-2).clamp_max(t - 1)
        bi = torch.arange(b, device=x.device).view(b, 1, 1, 1)
        hi = torch.arange(h, device=x.device).view(1, h, 1, 1)
        gk = k[bi, hi, tok]
        gv = v[bi, hi, tok]
        pos = torch.arange(t, device=x.device).view(1, 1, t, 1)
        future = tok > pos
        add = torch.where(future, torch.tensor(float("-inf"), device=x.device),
                          torch.tensor(0.0, device=x.device)).unsqueeze(-2)
        sc2 = (q.unsqueeze(-2).float() @ gk.transpose(-1, -2).float()) / math.sqrt(hd) + add
        y_sparse = (torch.softmax(sc2, dim=-1).to(x.dtype) @ gv).squeeze(-2)
        y = (y_local + y_sparse).transpose(1, 2).contiguous().view(b, t, h * hd)
        y = self.o(y)
        return torch.sigmoid(self.ogate(x)) * y


class DSv41(nn.Module):
    # deepseek-v4.1-flash lite cut. mhc-2 streams instead of residuals, csa
    # retrieval attention, hash-moe bootstrap on the first two mlps, routed
    # moe after, 3 mtp probes. deviations: static mhc mixing (real is input
    # dependent), hca folded into csa, engram hash approximated by token hash
    def __init__(self, cfg, n_exp=12, top_k=2):
        super().__init__()
        self.cfg = cfg
        self.future_scale = 1.0
        self.tok_emb = nn.Embedding(cfg.vocab, cfg.dim)
        self.blocks = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(cfg.layers):
            if i < 2:
                self.blocks.append(FlexAttention(cfg, out_gate=True, use_sink=True))
                self.mlps.append(TopKMoE(cfg, n_exp, top_k, exp_hidden=64, scoring="sqrtsoftplus",
                                         scaling=1.5, shared=True, swiglu_limit=10.0,
                                         frozen_router=True))
            else:
                self.blocks.append(CSALite(cfg))
                self.mlps.append(TopKMoE(cfg, n_exp, top_k, exp_hidden=64, scoring="sqrtsoftplus",
                                         scaling=1.5, shared=True, swiglu_limit=10.0))
        self.n1 = nn.ModuleList([RMSNorm(cfg.dim) for _ in range(cfg.layers)])
        self.n2 = nn.ModuleList([RMSNorm(cfg.dim) for _ in range(cfg.layers)])
        # mhc-lite: 2 streams, row-stochastic mixing, init equals plain residual
        self.A = nn.Parameter(torch.zeros(1, 2))
        self.B = nn.Parameter(torch.zeros(2, 2))
        self.C = nn.Parameter(torch.zeros(2, 1))
        self.norm = RMSNorm(cfg.dim)
        self.probes = nn.ModuleList([nn.Linear(cfg.dim, cfg.dim, bias=False) for _ in range(3)])

    def forward(self, idx, targets=None):
        b, t = idx.shape
        hd = self.cfg.dim // self.cfg.q_heads
        rd = int(max(int(hd * 0.25) // 2 * 2, 2))
        # two rope tables: full-dim for the dense first layers, quarter-dim
        # partial rope for the csa layers. both fp32 then cast with the stream
        cosf, sinf = build_rope(t, hd, self.cfg.rope_theta, idx.device)
        cosp, sinp = build_rope(t, rd, self.cfg.rope_theta, idx.device)
        e = self.tok_emb(idx)
        cosf, sinf = cosf.to(e.dtype), sinf.to(e.dtype)
        cosp, sinp = cosp.to(e.dtype), sinp.to(e.dtype)
        X = torch.stack([e, e], dim=-2)
        Am = torch.softmax(self.A, dim=-1)
        Bm = torch.softmax(self.B, dim=-1)
        Cm = 2 * torch.sigmoid(self.C)
        for attn, mlp, n1, n2 in zip(self.blocks, self.mlps, self.n1, self.n2):
            h_in = (Am @ X).squeeze(-2)
            if isinstance(attn, CSALite):
                o1 = attn(n1(h_in), cosp, sinp)
            else:
                o1 = attn(n1(h_in), cosf, sinf, window=48)
            X = Bm @ X + Cm * o1.unsqueeze(-2)
            h_in = (Am @ X).squeeze(-2)
            o2 = mlp(n2(h_in))
            X = Bm @ X + Cm * o2.unsqueeze(-2)
        h = self.norm((Am @ X).squeeze(-2))
        W = self.tok_emb.weight
        logits = F.linear(h, W)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            w = 0.3 * self.future_scale
            for i, probe in enumerate(self.probes, start=1):
                lf = F.linear(probe(h[:, :-i] if i < t else h), W)
                g = targets[:, i:]
                n = min(lf.size(1), g.size(1))
                loss = loss + w * F.cross_entropy(
                    lf[:, :n].reshape(-1, lf.size(-1)), g[:, :n].reshape(-1), reduction="mean")
        return logits, loss


def causal_dw_conv(x, conv):
    # depthwise short convolution on projections, kimi k4 silu recipe
    # left-pad k-1 so each step only sees the past, output length stays t
    # (trimming the right drops the last tokens, classic causal bug)
    k = conv.kernel_size[0] if isinstance(conv.kernel_size, tuple) else conv.kernel_size
    y = x.transpose(1, 2)
    y = conv(F.pad(y, (k - 1, 0)))
    return F.silu(y).transpose(1, 2)


class KimiDelta(GatedDeltaNet):
    # kda-lite: parent delta memory plus short convs on qkv and a gated
    # output norm. real kda runs low-rank gates with a-log decay inside
    # fused kernels, lite keeps per-head gates and documents the gap
    def __init__(self, cfg, kernel=4):
        super().__init__(cfg)
        d = cfg.dim
        self.q_conv = nn.Conv1d(d, d, kernel, groups=d, bias=False)
        self.k_conv = nn.Conv1d(d, d, kernel, groups=d, bias=False)
        self.v_conv = nn.Conv1d(d, d, kernel, groups=d, bias=False)
        self.onorm = RMSNorm(d)
        self.ogate = nn.Linear(d, d, bias=False)

    def forward(self, x, cos, sin, window=None):
        b, t, _ = x.shape
        h, hd, kv = self.q_heads, self.head_dim, self.kv_heads
        q = self.q(causal_dw_conv(x, self.q_conv)).view(b, t, h, hd).transpose(1, 2)
        k = self.k(causal_dw_conv(x, self.k_conv)).view(b, t, kv, hd).transpose(1, 2)
        v = self.v(causal_dw_conv(x, self.v_conv)).view(b, t, kv, hd).transpose(1, 2)
        if self.q_norm is not None:
            q = self.q_norm(q)
        q = apply_rope(q, cos, sin)
        k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
        g = torch.sigmoid(self.gate(x)).view(b, t, h, 1).transpose(1, 2)
        beta = torch.sigmoid(self.beta(x)).view(b, t, h, 1).transpose(1, 2)
        k = repeat_kv(k, self.rep)
        v = repeat_kv(v, self.rep)
        state = q.new_zeros(b, h, hd, hd)
        outs = []
        for i in range(t):
            qi, ki, vi = q[:, :, i], k[:, :, i], v[:, :, i]
            gi, bi = g[:, :, i], beta[:, :, i]
            pred = (state * ki.unsqueeze(-1)).sum(-2)
            state = state * gi.unsqueeze(-1) + bi.unsqueeze(-1) * (vi - pred).unsqueeze(-1) * ki.unsqueeze(-2)
            outs.append((state * qi.unsqueeze(-1)).sum(-2))
        y = torch.stack(outs, dim=2).transpose(1, 2).contiguous().view(b, t, h * hd)
        y = self.o(self.onorm(y) * torch.sigmoid(self.ogate(x)))
        return y


class KimiBackbone(nn.Module):
    # kimi-k3 lite cut. 3 kda layers per gated mla, attention residuals read
    # every layer off history appended every 3rd, latent moe everywhere past
    # layer 0, situ activations throughout. deviations: kda gate form is
    # per-head not low-rank, attnres block size 3 not 12, 64 latent experts
    # not 896, embeddings tied, no vision, no quant
    def __init__(self, cfg, res_every=3):
        super().__init__()
        self.cfg = cfg
        self.res_every = res_every
        self.tok_emb = nn.Embedding(cfg.vocab, cfg.dim)
        self.blocks = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(cfg.layers):
            full = (i + 1) % 4 == 0
            self.blocks.append(MLABlock(cfg) if full else KimiDeltaBlock(cfg))
            if i == 0:
                self.mlps.append(SituMLP(cfg))
            else:
                self.mlps.append(LatentMoE(cfg))
        self.n1 = nn.ModuleList([RMSNorm(cfg.dim) for _ in range(cfg.layers)])
        self.n2 = nn.ModuleList([RMSNorm(cfg.dim) for _ in range(cfg.layers)])
        self.ra_n = nn.ModuleList([RMSNorm(cfg.dim) for _ in range(cfg.layers)])
        self.ra_p = nn.ModuleList([nn.Linear(cfg.dim, 1, bias=False) for _ in range(cfg.layers)])
        self.rm_n = nn.ModuleList([RMSNorm(cfg.dim) for _ in range(cfg.layers)])
        self.rm_p = nn.ModuleList([nn.Linear(cfg.dim, 1, bias=False) for _ in range(cfg.layers)])
        self.norm = RMSNorm(cfg.dim)

    def forward(self, idx, targets=None):
        b, t = idx.shape
        cos, sin = build_rope(t, self.cfg.dim // self.cfg.q_heads, self.cfg.rope_theta, idx.device)
        s = self.tok_emb(idx)
        cos, sin = cos.to(s.dtype), sin.to(s.dtype)
        hist = s.new_zeros(b * t, 0, s.size(-1))
        for i, (blk, mlp, n1, n2, ran, rap, rmn, rmp) in enumerate(
                zip(self.blocks, self.mlps, self.n1, self.n2,
                    self.ra_n, self.ra_p, self.rm_n, self.rm_p)):
            if hist.size(1) > 0:
                s = apply_attn_res(s.reshape(-1, s.size(-1)), hist, rap.weight, ran.weight)
                s = s.view(b, t, -1)
            if i % self.res_every == 0:
                hist = torch.cat([hist, s.detach().reshape(-1, s.size(-1)).unsqueeze(1)], dim=1)
            s = s + blk(n1(s), cos, sin)
            flat = s.reshape(-1, s.size(-1))
            if hist.size(1) > 0:
                flat = apply_attn_res(flat, hist, rmp.weight, rmn.weight)
            s = s + mlp(n2(flat.view(b, t, -1)))
        h = self.norm(s)
        logits = F.linear(h, self.tok_emb.weight)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


class KimiDeltaBlock(nn.Module):
    def __init__(self, cfg, sliding=None):
        super().__init__()
        self.n1 = RMSNorm(cfg.dim)
        self.n2 = RMSNorm(cfg.dim)
        self.attn = KimiDelta(cfg)
        self.mlp = None

    def forward(self, x, cos, sin):
        # mlp lives in the backbone (latent moe), block is attention only
        # n2 kept so optimizer groups match the family shape
        return self.attn(self.n1(x), cos, sin)


class OSS120(nn.Module):
    # gpt-oss-120b lite cut. alternating sliding and full attention with sink
    # scalars, bias linears, exact interleaved expert form with clamps,
    # top-4 of 8 sigmoid-free router (softmax over topk), yarn rope.
    # deviations: 8 experts not 128, bias-balance not 0.9 aux loss, tied head
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab, cfg.dim)
        self.blocks = nn.ModuleList()
        for i in range(cfg.layers):
            win = 64 if i % 2 == 0 else None
            self.blocks.append(OSS120Block(cfg, window=win))
        self.norm = RMSNorm(cfg.dim, eps=1e-5)

    def forward(self, idx, targets=None):
        b, t = idx.shape
        cos, sin = yarn_tables(t, self.cfg.dim // self.cfg.q_heads, 150000.0,
                               32.0, 4096, device=idx.device)
        x = self.tok_emb(idx)
        cos, sin = cos.to(x.dtype), sin.to(x.dtype)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        h = self.norm(x)
        logits = F.linear(h, self.tok_emb.weight)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


class OSS120Block(nn.Module):
    def __init__(self, cfg, window=None):
        super().__init__()
        self.n1 = RMSNorm(cfg.dim, eps=1e-5)
        self.n2 = RMSNorm(cfg.dim, eps=1e-5)
        self.attn = FlexAttention(cfg, bias=True, use_sink=True, eps=1e-5)
        self.mlp = TopKMoE(cfg, n_exp=8, top_k=4, exp_hidden=192, scoring="oss-topk",
                           norm_topk=False, scaling=1.0, shared=False,
                           oss_style=True, bias_terms=True)
        self.window = window

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin, window=self.window)
        x = x + self.mlp(self.n2(x))
        return x


class GLM53(nn.Module):
    # glm-5.3 lite cut. mla attention with a dsa indexer every 4th layer,
    # shared reuse between, dense mlps on the first 3 layers, routed moe
    # after, 1 mtp head. deviations: indexer 4 heads not 32, topk 32 not
    # 2048, no yarn (config rope is default type), tied head
    def __init__(self, cfg, n_exp=12, top_k=2):
        super().__init__()
        self.cfg = cfg
        self.future_scale = 1.0
        self.tok_emb = nn.Embedding(cfg.vocab, cfg.dim)
        self.blocks = nn.ModuleList([GLMBlock(cfg) for _ in range(cfg.layers)])
        self.indexers = nn.ModuleList(
            [GLMIndexer(cfg) if i % 4 == 3 else None for i in range(cfg.layers)])
        self.n1 = nn.ModuleList([RMSNorm(cfg.dim) for _ in range(cfg.layers)])
        self.n2 = nn.ModuleList([RMSNorm(cfg.dim) for _ in range(cfg.layers)])
        self.norm = RMSNorm(cfg.dim)
        self.probe = nn.Linear(cfg.dim, cfg.dim, bias=False)

    def forward(self, idx, targets=None):
        b, t = idx.shape
        cos, sin = build_rope(t, self.cfg.dim // self.cfg.q_heads, self.cfg.rope_theta, idx.device)
        x = self.tok_emb(idx)
        cos, sin = cos.to(x.dtype), sin.to(x.dtype)
        from model import causal_window_mask
        base = causal_window_mask(t, None, x.device).to(x.dtype)
        prev_topk = None
        for i, (blk, n1, n2, indexer) in enumerate(zip(self.blocks, self.n1, self.n2, self.indexers)):
            if indexer is not None:
                prev_topk = indexer(x, cos, sin)
            mask = base
            if prev_topk is not None:
                sel = torch.zeros(b, t, t, dtype=torch.bool, device=x.device)
                sel.scatter_(-1, prev_topk.clamp_max(t - 1), True)
                # head dim of ones so sdpa broadcasts over query heads
                mask = (base + torch.where(sel, 0.0, torch.tensor(float("-inf"), device=x.device))).unsqueeze(1)
            x = x + blk.attn(n1(x), cos, sin, mask=mask)
            if i < 3:
                x = x + blk.mlp_dense(n2(x))
            else:
                x = x + blk.mlp_moe(n2(x))
        h = self.norm(x)
        W = self.tok_emb.weight
        logits = F.linear(h, W)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            if t > 1:
                lf = F.linear(self.probe(h[:, :-1]), W)
                n = min(lf.size(1), targets.size(1) - 1)
                loss = loss + 0.3 * self.future_scale * F.cross_entropy(
                    lf[:, :n].reshape(-1, lf.size(-1)),
                    targets[:, 1:n + 1].reshape(-1), reduction="mean")
        return logits, loss


class GLMBlock(nn.Module):
    def __init__(self, cfg, sliding=None):
        super().__init__()
        self.n1 = RMSNorm(cfg.dim)
        self.n2 = RMSNorm(cfg.dim)
        self.attn = SparseMLA(cfg)
        self.mlp_dense = SwiGLU(cfg)
        self.mlp_moe = TopKMoE(cfg, n_exp=12, top_k=2, exp_hidden=64, scoring="sigmoid",
                               scaling=2.5, shared=True)

    def forward(self, x, cos, sin):
        # backbone drives (mask lives there), direct calls pass through dense
        return x


class SparseMLA(MLAAttention):
    # mla body with an additive sparse mask hook for indexer topk
    def forward(self, x, cos, sin, window=None, mask=None):
        b, t, _ = x.shape
        h, hd = self.q_heads, self.head_dim
        q = self.q(x).view(b, t, h, hd).transpose(1, 2)
        kl = self.k_norm(self.k_down(x).view(b, t, self.kv_heads, self.lat))
        vl = self.v_norm(self.v_down(x).view(b, t, self.kv_heads, self.lat))
        k = self.k_up(kl.flatten(2)).view(b, t, self.kv_heads, hd).transpose(1, 2)
        v = self.v_up(vl.flatten(2)).view(b, t, self.kv_heads, hd).transpose(1, 2)
        if self.q_norm is not None:
            q = self.q_norm(q)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        k = repeat_kv(k, self.rep)
        v = repeat_kv(v, self.rep)
        from model import causal_window_mask
        m = causal_window_mask(t, window, x.device).to(x.dtype)
        if mask is not None:
            m = m + mask.to(x.dtype)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
        return self.o(y.transpose(1, 2).contiguous().view(b, t, h * hd))


class GLMIndexer(nn.Module):
    # dsa indexer lite: own tiny q/k projs, relu scores, head weighting,
    # topk 32. real one runs 32 heads and topk 2048 at 1m context
    def __init__(self, cfg, heads=4, head_dim=32, topk=32):
        super().__init__()
        self.heads = heads
        self.topk = topk
        self.scale = head_dim ** -0.5
        self.wq = nn.Linear(cfg.dim, heads * head_dim, bias=False)
        self.wk = nn.Linear(cfg.dim, head_dim, bias=False)
        self.knorm = nn.LayerNorm(head_dim)
        self.wproj = nn.Linear(cfg.dim, heads, bias=False)

    def forward(self, x, cos, sin):
        b, t, _ = x.shape
        q = self.wq(x).view(b, t, self.heads, -1)
        k = self.knorm(self.wk(x))
        scores = torch.relu((q.float() @ k.transpose(-1, -2).unsqueeze(1).float())) * self.scale
        w = self.wproj(x.float()) * (self.heads ** -0.5)
        idx_scores = (w.unsqueeze(-2) @ scores).squeeze(-2)
        causal = torch.arange(t, device=x.device).unsqueeze(0) >= torch.arange(t, device=x.device).unsqueeze(1)
        idx_scores = idx_scores.masked_fill(~causal, float("-inf"))
        return idx_scores.topk(min(self.topk, t), dim=-1).indices.to(torch.int32)


class Gemma4(nn.Module):
    # gemma-4 lite cut. 5 sliding layers per full one, sliding rope theta
    # 10k and full rope theta 1m on separate tables, logit softcap 30,
    # tanh gelu mlps, tied head. deviations: window 64 not 1024, kv counts
    # shared across layer types, no vision, embeddings tied (config agrees)
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab, cfg.dim)
        self.blocks = nn.ModuleList()
        for i in range(cfg.layers):
            full = (i + 1) % 6 == 0
            self.blocks.append(GemmaBlock(cfg, full=full))
        self.norm = RMSNorm(cfg.dim)

    def forward(self, idx, targets=None):
        b, t = idx.shape
        hd = self.cfg.dim // self.cfg.q_heads
        coss, sins = build_rope(t, hd, 10000.0, idx.device)
        cosf, sinf = build_rope(t, hd, 1000000.0, idx.device)
        x = self.tok_emb(idx)
        coss, sins = coss.to(x.dtype), sins.to(x.dtype)
        cosf, sinf = cosf.to(x.dtype), sinf.to(x.dtype)
        for blk in self.blocks:
            x = blk(x, (coss, sins) if not blk.full else (cosf, sinf))
        h = self.norm(x)
        logits = 30.0 * torch.tanh(F.linear(h.float(), self.tok_emb.weight.float()) / 30.0)
        logits = logits.to(x.dtype)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


class GemmaBlock(nn.Module):
    def __init__(self, cfg, full=False):
        super().__init__()
        self.n1 = RMSNorm(cfg.dim)
        self.n2 = RMSNorm(cfg.dim)
        self.attn = FlexAttention(cfg)
        self.mlp = ActMLP(cfg, act="tanh_gelu")
        self.full = full
        self.window = None if full else 64

    def forward(self, x, tables):
        cos, sin = tables
        x = x + self.attn(self.n1(x), cos, sin, window=self.window)
        x = x + self.mlp(self.n2(x))
        return x


BUILDERS = [b_base, b_qknorm, b_combo, b_remy,
            b_remy_ls, b_remy_post, b_remy_sand, b_qk_post, b_qk_ls,
            b_parallel, b_parallel_qk, b_remy_parallel,
            b_geglu, b_reglu, b_mqa, b_mha, b_wide, b_narrow,
             b_mla, b_gated, b_remy_mtp, b_layerdrop, b_looped, b_qwen35,
             b_dsv41, b_oss120, b_kimi, b_glm, b_gemma, b_remy_grind,
             b_step5, b_inkling, b_nemotron, b_liquid, b_granite]


# one-liners for the final report. what it is and why it might dethrone remy
DESCRIBE = {
    "base": "plain llama bones, the skeleton everything must beat",
    "qknorm": "qwen-style qk norm, 200-step champ on stability",
    "combo": "qknorm plus interleaved sliding windows, cheap long range",
    "remy": "differential subtractive attention, 1k-step champ, the one to beat",
    "remy-ls": "remy plus layerscale gates, denoise with a volume knob",
    "remy-post": "remy plus normformer scrubs, cleanest stream in the zoo",
    "remy-sand": "remy plus sandwich norms, maximum signal control",
    "qk-post": "qknorm plus normformer, stability stacked on stability",
    "qk-ls": "qknorm plus layerscale, gentle branches",
    "parallel": "palm wiring, attn and mlp side by side, half the depth",
    "parallel-qk": "palm wiring with qknorm, systems win meets stability win",
    "remy-parallel": "champ attention in palm wiring, both wins at once",
    "geglu": "gpt-2 activation in a llama body, old religion check",
    "reglu": "relu gating, sparser mlps, small-model folk tale",
    "mqa": "one kv head for all queries, cache slashed, params refilled nowhere",
    "mha8": "every query its own kv head, max expressivity per flop",
    "wide-shallow": "3 fat layers, width religion at fixed budget",
    "narrow-deep": "12 thin layers, depth religion at fixed budget",
    "mla-lite": "deepseek latent kv, half the cache, honesty enforced by the bench",
    "gated": "per-token output gates, heads that refuse to emit noise",
    "remy-mtp": "champ plus futures, signal times signal",
    "layerdrop": "random whole-layer skips in training, free ensemble at inference",
    "looped-act": "one shared block looped 8x with act halting, depth as compute",
    "qwen35": "real qwen3.5 recipe: 3 deltanet per gated attention, output gates, 1 mtp",
    "deepseek": "v3 lineage flavor: mla cache plus 8-expert moe with shared expert, aux-free",
    "gpt-oss": "openai open flavor: sliding gqa plus 8-expert moe, no shared",
    "ds-v41": "v4.1-flash faithful: csa retrieval plus mhc-2 streams plus hash-moe plus 3 mtp",
    "oss-120": "120b faithful: alternate sliding-full plus sink plus exact clamped experts plus yarn",
    "kimi-k3": "k3 faithful: 3 kda per mla plus attres plus situ plus latent-64 moe plus 2 shared",
    "glm-53": "53 faithful: mla plus 3-1 shared-full indexer plus dense-first plus mtp-1",
    "gemma-4": "4 faithful: 5-1 sliding-full plus dual rope plus softcap 30 plus tanh-gelu",
    "remy-grind": "98k speed recipe: plain attn, no qk/rope, layernorm. zoo judges its brain",
    "step-5": "stepfun narrow-deep: 3 sliding plus full with csa sparse mask, clamped moe, 1 mtp",
    "inkling": "thinking-machines: sliding/global interleave, 8-exp top-2 plus shared moe",
    "nemotron-mamba": "nvidia hybrid: mamba-2 selective scan plus moe-only plus attention every 6th, 1 mtp",
    "liquid-conv": "liquid conv-dominant: 2 short-conv blocks per full attention, theta 10m",
    "granite-40": "ibm narrow-deep dense: 20 layers, theta 10m, tied head",
}


if __name__ == "__main__":
    # scorecard before any gpu burns. every entry should read ok near 50m
    report_table([(n, m) for n, m in (b() for b in BUILDERS)])
