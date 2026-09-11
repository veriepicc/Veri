import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# mini gpt from scratch, ~2m params, built to settle optimizer bets
# no hf, no trainer, plain pytorch you can read in one sitting


class Block(nn.Module):
    def __init__(self, dim, heads, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        b, t, c = x.shape
        h = self.heads
        q, k, v = self.qkv(self.ln1(x)).split(c, dim=2)
        q = q.view(b, t, h, c // h).transpose(1, 2)
        k = k.view(b, t, h, c // h).transpose(1, 2)
        v = v.view(b, t, h, c // h).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        x = x + self.drop(self.proj(y))
        x = x + self.drop(self.mlp(self.ln2(x)))
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab=50257, layers=3, heads=4, dim=128, seq=96, dropout=0.0):
        super().__init__()
        self.seq = seq
        self.tok_emb = nn.Embedding(vocab, dim)
        self.pos_emb = nn.Embedding(seq, dim)
        self.blocks = nn.ModuleList([Block(dim, heads, dropout) for _ in range(layers)])
        self.ln = nn.LayerNorm(dim)
        # output head tied to embeddings, half the params for free
        self.apply(self._init)

    def _init(self, m):
        # plain init, scaled down on residual outputs so deep stacks behave
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        b, t = idx.shape
        pos = torch.arange(t, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for blk in self.blocks:
            x = blk(x)
        logits = F.linear(self.ln(x), self.tok_emb.weight)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def count(self):
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    m = MiniGPT()
    print(f"params: {m.count() / 1e6:.2f}M")
    x = torch.randint(0, 50257, (2, m.seq))
    _, loss = m(x, x)
    print(f"smoke loss: {loss.item():.3f}")
