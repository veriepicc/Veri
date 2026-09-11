import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import VeriConfig, RMSNorm, build_rope, apply_rope, repeat_kv, SwiGLU, causal_window_mask


class RemyAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.q_heads = cfg.q_heads
        self.kv_heads = cfg.kv_heads
        self.head_dim = cfg.dim // cfg.q_heads
        self.rep = cfg.q_heads // cfg.kv_heads
        assert cfg.q_heads % 2 == 0
        self.q = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.k = nn.Linear(cfg.dim, cfg.kv_heads * self.head_dim, bias=False)
        self.v = nn.Linear(cfg.dim, cfg.kv_heads * self.head_dim, bias=False)
        self.o = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim) if cfg.qk_norm else None
        self.k_norm = RMSNorm(self.head_dim) if cfg.qk_norm else None
        self.lam = nn.Parameter(torch.zeros(cfg.q_heads // 2) + 0.5)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cos, sin, window=None):
        b, t, _ = x.shape
        h, hd = self.q_heads, self.head_dim
        q = self.q(x).view(b, t, h, hd).transpose(1, 2)
        k = self.k(x).view(b, t, self.kv_heads, hd).transpose(1, 2)
        v = self.v(x).view(b, t, self.kv_heads, hd).transpose(1, 2)
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        k = repeat_kv(k, self.rep)
        v = repeat_kv(v, self.rep)
        mask = causal_window_mask(t, window, x.device).to(x.dtype)
        h2 = h // 2
        y1 = F.scaled_dot_product_attention(q[:, :h2], k[:, :h2], v[:, :h2], attn_mask=mask)
        y2 = F.scaled_dot_product_attention(q[:, h2:], k[:, h2:], v[:, h2:], attn_mask=mask)
        lam = self.lam.sigmoid().view(1, h2, 1, 1)
        p1 = y1 - lam * y2
        y = torch.cat([p1, y2], dim=1)
        y = y.transpose(1, 2).contiguous().view(b, t, h * hd)
        return self.drop(self.o(y))


class RemyBlock(nn.Module):
    def __init__(self, cfg, sliding=None):
        super().__init__()
        self.n1 = RMSNorm(cfg.dim)
        self.n2 = RMSNorm(cfg.dim)
        self.attn = RemyAttention(cfg)
        self.mlp = SwiGLU(cfg)
        self.sliding = sliding

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin, self.sliding)
        x = x + self.mlp(self.n2(x))
        return x


class RemyBackbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab, cfg.dim)
        self.pos_drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            [RemyBlock(cfg, sliding=cfg.sliding if i % 2 == 0 else None) for i in range(cfg.layers)]
        )
        self.norm = RMSNorm(cfg.dim)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02 / math.sqrt(2 * self.cfg.layers))
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        b, t = idx.shape
        cos, sin = build_rope(t, self.cfg.dim // self.cfg.q_heads, self.cfg.rope_theta, idx.device)
        x = self.pos_drop(self.tok_emb(idx))
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
