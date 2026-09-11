import math
import torch


TERRY_VERSION = "2.2-momdtype"


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


def match_rms(Z, target):
    rms = Z.float().norm() / math.sqrt(Z.numel())
    return Z * (target / (rms + 1e-12))


class Terry:
    def __init__(self, lr=0.01, beta=0.95, wd=0.01, ns_steps=5, target_rms=0.2, muon_max_dim=8192, nesterov=True, warmup_steps=100, nesterov_early=False, table_mode="sign", table_beta2=0.99, table_eps=1e-8, table_nesterov=False, matrix_mode="ns", table_beta=None):
        self.lr = lr
        self.beta = beta
        self.wd = wd
        self.ns_steps = ns_steps
        self.target_rms = target_rms
        self.muon_max_dim = muon_max_dim
        self.nesterov = nesterov
        self.nesterov_early = nesterov_early
        self.table_mode = table_mode
        self.table_beta2 = table_beta2
        self.table_eps = table_eps
        self.table_nesterov = table_nesterov
        self.matrix_mode = matrix_mode
        self.table_beta = table_beta if table_beta is not None else beta
        self._tab = {}
        self._mat = {}
        self.warmup_steps = warmup_steps
        self.step_count = 0
        self.params = []
        self.state = {}
        self.scales = {}

    def add_param_group(self, params, scale=1.0):
        for p in params:
            if p.requires_grad and id(p) not in self.state:
                self.params.append(p)
                self.state[id(p)] = torch.zeros(p.numel(), dtype=p.dtype, device=p.device).view(p.shape)
                self.scales[id(p)] = scale

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    def _is_matrix(self, p):
        return (
            p.dim() == 2
            and p.size(0) <= self.muon_max_dim
            and p.size(1) <= self.muon_max_dim
        )

    @torch.no_grad()
    def step(self):
        b = self.beta
        self.step_count += 1
        t = 1.0 if self.warmup_steps <= 0 else min(1.0, self.step_count / self.warmup_steps)
        eff_rms = self.target_rms * (0.2 + 0.8 * t)
        eff_ns = max(2, round(self.ns_steps * t))
        for p in self.params:
            g = p.grad
            if g is None:
                continue
            m = self.state[id(p)]
            gf = g.float() if g.dtype != torch.float32 else g
            gm = gf.to(m.dtype)
            is_mat = self._is_matrix(p)
            bb = b if is_mat else self.table_beta
            m.mul_(bb).add_(gm, alpha=1.0 - bb)
            if is_mat:
                if self.nesterov and (t >= 1.0 or self.nesterov_early):
                    d = m.clone().mul_(bb).add_(gm)
                elif self.nesterov:
                    d = m.clone().mul_(bb * t).add_(gm)
                else:
                    d = m
                if self.matrix_mode in ("graft", "graft-l") and p.dim() >= 2:
                    st = self._mat.get(id(p))
                    if st is None:
                        st = {"r": torch.zeros(p.size(0), dtype=torch.float32, device=p.device),
                              "c": torch.zeros(p.size(1), dtype=torch.float32, device=p.device)
                              if self.matrix_mode == "graft" else None}
                        self._mat[id(p)] = st
                    g2 = gf.to(m.device).float().square()
                    st["r"].lerp_(g2.mean(dim=1), 1.0 - self.table_beta2)
                    if st["c"] is not None:
                        st["c"].lerp_(g2.mean(dim=0), 1.0 - self.table_beta2)
                        v = st["r"].unsqueeze(1) * st["c"].unsqueeze(0)
                        v = v / st["c"].mean().clamp_min(1e-12)
                        d = d / (v.sqrt() + self.table_eps)
                    else:
                        rs = (st["r"] / st["r"].mean().clamp_min(1e-12)).sqrt().unsqueeze(1)
                        d = d / (rs + self.table_eps)
                update = match_rms(newton_schulz(d, eff_ns), eff_rms)
            else:
                mt = m
                if self.table_nesterov:
                    mt = m.clone().mul_(bb).add_(gm)
                if self.table_mode == "factored" and p.dim() >= 2:
                    st = self._tab.get(id(p))
                    if st is None:
                        st = {"r": torch.zeros(p.size(0), dtype=torch.float32, device=p.device),
                              "c": torch.zeros(p.size(1), dtype=torch.float32, device=p.device)}
                        self._tab[id(p)] = st
                    b2 = self.table_beta2
                    g2 = gf.to(m.device).float().square()
                    st["r"].lerp_(g2.mean(dim=1), 1.0 - b2)
                    st["c"].lerp_(g2.mean(dim=0), 1.0 - b2)
                    v = st["r"].unsqueeze(1) * st["c"].unsqueeze(0)
                    v = v / st["c"].mean().clamp_min(1e-12)
                    u = mt / (v.sqrt() + self.table_eps)
                    update = match_rms(u, eff_rms)
                elif self.table_mode == "factored":
                    st = self._tab.get(id(p))
                    if st is None:
                        st = {"r": torch.zeros(p.numel(), dtype=torch.float32, device=p.device),
                              "c": None}
                        self._tab[id(p)] = st
                    st["r"].lerp_(gf.to(m.device).float().square().reshape(-1), 1.0 - self.table_beta2)
                    u = mt.reshape(-1) / (st["r"].sqrt() + self.table_eps)
                    update = match_rms(u.reshape(p.shape), eff_rms)
                else:
                    update = torch.sign(mt) * eff_rms
            scale = self.scales[id(p)]
            if self.wd != 0.0 and p.dim() >= 2:
                p.data.mul_(1.0 - self.lr * self.wd)
            p.data.add_(update.to(p.dtype), alpha=-self.lr * scale)

    def state_dict(self):
        return {
            "lr": self.lr,
            "beta": self.beta,
            "wd": self.wd,
            "ns_steps": self.ns_steps,
            "target_rms": self.target_rms,
            "nesterov": self.nesterov,
            "nesterov_early": self.nesterov_early,
            "warmup_steps": self.warmup_steps,
            "step_count": self.step_count,
            "table_mode": self.table_mode,
            "table_beta2": self.table_beta2,
            "table_eps": self.table_eps,
            "table_nesterov": self.table_nesterov,
            "matrix_mode": self.matrix_mode,
            "table_beta": self.table_beta,
            "scales": [self.scales[id(p)] for p in self.params],
            "moments": [self.state[id(p)].clone() for p in self.params],
            "tab": [(i, self._tab[id(p)]["r"].clone(),
                     self._tab[id(p)]["c"].clone() if self._tab[id(p)]["c"] is not None else None)
                    for i, p in enumerate(self.params) if id(p) in self._tab],
            "mat": [(i, self._mat[id(p)]["r"].clone(),
                     self._mat[id(p)]["c"].clone() if self._mat[id(p)]["c"] is not None else None)
                    for i, p in enumerate(self.params) if id(p) in self._mat],
        }

    def load_state_dict(self, params, d):
        self.params = []
        self.state = {}
        self.scales = {}
        self.add_param_group(params)
        self.lr = d["lr"]
        self.beta = d["beta"]
        self.wd = d["wd"]
        self.ns_steps = d["ns_steps"]
        self.target_rms = d["target_rms"]
        self.nesterov = d.get("nesterov", True)
        self.nesterov_early = d.get("nesterov_early", False)
        self.warmup_steps = d.get("warmup_steps", 100)
        self.step_count = d.get("step_count", 0)
        self.table_mode = d.get("table_mode", "sign")
        self.table_beta2 = d.get("table_beta2", 0.99)
        self.table_eps = d.get("table_eps", 1e-8)
        self.table_nesterov = d.get("table_nesterov", False)
        self.matrix_mode = d.get("matrix_mode", "ns")
        self.table_beta = d.get("table_beta", None)
        if self.table_beta is None:
            self.table_beta = self.beta
        self.restart = d.get("restart", False)
        self._tab = {}
        for i, r, c in d.get("tab", []):
            p = self.params[i]
            self._tab[id(p)] = {"r": r.to(p.device).clone(),
                                "c": c.to(p.device).clone() if c is not None else None}
        self._mat = {}
        for i, r, c in d.get("mat", []):
            p = self.params[i]
            self._mat[id(p)] = {"r": r.to(p.device).clone(),
                                "c": c.to(p.device).clone() if c is not None else None}
        scales = d.get("scales", [1.0] * len(self.params))
        for p, m, s in zip(self.params, d["moments"], scales):
            self.state[id(p)].copy_(m)
            self.scales[id(p)] = s


def cosine_schedule(step, total, peak, floor=0.0, warmup=0.0):
    if warmup > 0 and step < warmup:
        return peak * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    t = min(1.0, max(0.0, t))
    return floor + 0.5 * (peak - floor) * (1.0 + math.cos(math.pi * t))
