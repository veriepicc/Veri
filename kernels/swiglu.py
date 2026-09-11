import torch
import torch.nn.functional as F

# swiglu fused — silu(gate) * up in one pass, bwd in one pass
# narrowed views (no i32 overflow), BLOCK=1024 + LOOP=4 (register fit)
# dynamo-opaque: compiler calls it eager, no tracing into triton dynamism

try:
    import triton
    import triton.language as tl

    @triton.jit
    def _swiglu_fwd_kernel(gate_ptr, up_ptr, out_ptr, count, BLOCK: tl.constexpr, LOOP: tl.constexpr):
        for i in range(LOOP):
            offs = (tl.program_id(0) * LOOP + i) * BLOCK + tl.arange(0, BLOCK)
            mask = offs < count
            g = tl.load(gate_ptr + offs, mask=mask).to(tl.float32)
            u = tl.load(up_ptr + offs, mask=mask).to(tl.float32)
            s = g * tl.sigmoid(g)
            tl.store(out_ptr + offs, (s * u).to(out_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _swiglu_bwd_kernel(gate_ptr, up_ptr, grad_out_ptr, grad_gate_ptr, grad_up_ptr, count, BLOCK: tl.constexpr, LOOP: tl.constexpr):
        for i in range(LOOP):
            offs = (tl.program_id(0) * LOOP + i) * BLOCK + tl.arange(0, BLOCK)
            mask = offs < count
            g = tl.load(gate_ptr + offs, mask=mask).to(tl.float32)
            u = tl.load(up_ptr + offs, mask=mask).to(tl.float32)
            go = tl.load(grad_out_ptr + offs, mask=mask).to(tl.float32)
            sig = tl.sigmoid(g)
            silu = g * sig
            dsilu = sig * (1.0 + g * (1.0 - sig))
            gg = go * u * dsilu
            gu = go * silu
            tl.store(grad_gate_ptr + offs, gg.to(grad_gate_ptr.dtype.element_ty), mask=mask)
            tl.store(grad_up_ptr + offs, gu.to(grad_up_ptr.dtype.element_ty), mask=mask)

    HAS_TRITON = True
except Exception:
    HAS_TRITON = False


class _SwiGLUFused(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up):
        ctx.save_for_backward(gate, up)
        if HAS_TRITON and gate.is_cuda:
            out = torch.empty_like(gate)
            n = gate.numel()
            BLOCK, LOOP, CHUNK = 1024, 4, 1 << 30
            gf, uf, of = gate.reshape(-1), up.reshape(-1), out.reshape(-1)
            for base in range(0, n, CHUNK):
                count = min(CHUNK, n - base)
                grid = ((count + BLOCK * LOOP - 1) // (BLOCK * LOOP),)
                _swiglu_fwd_kernel[grid](gf[base:base + count], uf[base:base + count],
                                         of[base:base + count], count, BLOCK=BLOCK, LOOP=LOOP)
            return out
        else:
            return F.silu(gate) * up

    @staticmethod
    def backward(ctx, grad_out):
        gate, up = ctx.saved_tensors
        if HAS_TRITON and gate.is_cuda:
            gg = torch.empty_like(gate)
            gu = torch.empty_like(up)
            n = gate.numel()
            BLOCK, LOOP, CHUNK = 1024, 4, 1 << 30
            gf, uf, rf, af, bf = gate.reshape(-1), up.reshape(-1), grad_out.reshape(-1), gg.reshape(-1), gu.reshape(-1)
            for base in range(0, n, CHUNK):
                count = min(CHUNK, n - base)
                grid = ((count + BLOCK * LOOP - 1) // (BLOCK * LOOP),)
                _swiglu_bwd_kernel[grid](gf[base:base + count], uf[base:base + count],
                                         rf[base:base + count], af[base:base + count],
                                         bf[base:base + count], count, BLOCK=BLOCK, LOOP=LOOP)
            return gg, gu
        else:
            sig = torch.sigmoid(gate)
            silu = gate * sig
            dsilu = sig * (1 + gate * (1 - sig))
            gg = grad_out * up * dsilu
            gu = grad_out * silu
            return gg, gu


try:
    _DISABLE = torch.compiler.disable
except AttributeError:
    _DISABLE = torch._dynamo.disable


@_DISABLE
def swiglu_fused(gate, up):
    return _SwiGLUFused.apply(gate, up)
