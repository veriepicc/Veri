import math
import torch


# faithful muon, single device, no dist
# ported from keller jordan reference, aspect scaling plus nesterov included
# old version used moonlight dim scaling and skipped nesterov, that is gone
# hidden matrices get orthogonalized momentum, tables plus norms get aux adam
# embed path sign is ablation only, lets muon borrow terry trick for one run


def newton_schulz(G, steps=5):
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.float()
    transposed = False
    if X.size(0) > X.size(1):
        X = X.T
        transposed = True
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


def muon_update(g, m, beta=0.95, ns_steps=5, nesterov=True):
    # reference order, momentum first then lookahead then orthogonalize
    m.lerp_(g, 1.0 - beta)
    d = g.clone().lerp_(m, beta) if nesterov else m
    Z = newton_schulz(d, ns_steps)
    # aspect scaling from reference, tall matrices step a bit bigger
    Z = Z * max(1.0, Z.size(-2) / max(1, Z.size(-1))) ** 0.5
    return Z


def adam_update(g, b1, b2, step, betas, eps):
    b1.lerp_(g, 1.0 - betas[0])
    b2.lerp_(g.square(), 1.0 - betas[1])
    c1 = b1 / (1.0 - betas[0] ** step)
    c2 = b2 / (1.0 - betas[1] ** step)
    return c1 / (c2.sqrt() + eps)


class _AdamShim:
    # compat face for old harnesses, exposes param groups plus state view
    def __init__(self, host):
        self._host = host

    @property
    def param_groups(self):
        return self._host._adam_pg

    @property
    def state(self):
        out = {}
        for pid, st in self._host.adam_state.items():
            p = self._host._id2p.get(pid)
            if p is not None:
                out[p] = {"exp_avg": st["b1"], "exp_avg_sq": st["b2"]}
        return out

    def step(self):
        pass


class Muon:
    def __init__(self, lr=0.02, beta=0.95, wd=0.01, ns_steps=5, adam_lr=3e-4,
                 muon_max_dim=8192, nesterov=True, adam_betas=(0.9, 0.95),
                 adam_eps=1e-10, embed_rms=0.2, embed_path="adam"):
        # lr is matrix step, adam_lr is fallback step for tables and norms
        # nesterov defaults true now, matches reference, old code missed it
        # embed path adam is faithful, sign is hybrid ablation vs terry
        self.lr = lr
        self.beta = beta
        self.wd = wd
        self.ns_steps = ns_steps
        self.muon_max_dim = muon_max_dim
        self.nesterov = nesterov
        self.adam_betas = adam_betas
        self.adam_eps = adam_eps
        self.embed_rms = embed_rms
        self.embed_path = embed_path
        self.params = []
        self.state = {}
        self.adam_state = {}
        self._id2p = {}
        self._adam_params = []
        self._adam_pg = [{"params": self._adam_params, "lr": adam_lr}]
        self.adam = _AdamShim(self)

    @property
    def adam_lr(self):
        return self._adam_pg[0]["lr"]

    @adam_lr.setter
    def adam_lr(self, v):
        self._adam_pg[0]["lr"] = v

    def add_param_group(self, params):
        for p in params:
            if p.requires_grad and id(p) not in self.state:
                self.params.append(p)
                self.state[id(p)] = torch.zeros_like(p.data)
                self._id2p[id(p)] = p
                if not self._is_matrix(p):
                    self._adam_params.append(p)

    def _is_matrix(self, p):
        return (
            p.dim() == 2
            and p.size(0) <= self.muon_max_dim
            and p.size(1) <= self.muon_max_dim
        )

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    def state_bytes(self):
        # all live state, muon buffers plus lazy adam buffers, no hiding
        total = sum(v.numel() * v.element_size() for v in self.state.values())
        for st in self.adam_state.values():
            total += st["b1"].numel() * st["b1"].element_size()
            total += st["b2"].numel() * st["b2"].element_size()
        return total

    @torch.no_grad()
    def step(self):
        b = self.beta
        for p in self.params:
            g = p.grad
            if g is None:
                continue
            if self._is_matrix(p):
                m = self.state[id(p)]
                Z = muon_update(g, m, b, self.ns_steps, self.nesterov)
                if self.wd != 0.0:
                    p.data.mul_(1.0 - self.lr * self.wd)
                p.data.add_(Z.to(p.dtype), alpha=-self.lr)
            elif self.embed_path == "sign":
                # ablation lane, terry style sign momentum on tables
                m = self.state[id(p)]
                m.mul_(b).add_(g, alpha=1.0 - b)
                if self.wd != 0.0 and p.dim() >= 2:
                    p.data.mul_(1.0 - self.lr * self.wd)
                p.data.add_(torch.sign(m).to(p.dtype), alpha=-self.lr * self.embed_rms)
            else:
                st = self.adam_state.get(id(p))
                if st is None:
                    st = {"b1": torch.zeros_like(p.data), "b2": torch.zeros_like(p.data), "step": 0}
                    self.adam_state[id(p)] = st
                st["step"] += 1
                u = adam_update(g, st["b1"], st["b2"], st["step"], self.adam_betas, self.adam_eps)
                if self.wd != 0.0:
                    p.data.mul_(1.0 - self.adam_lr * self.wd)
                p.data.add_(u.to(p.dtype), alpha=-self.adam_lr)
