"""
dit_blocks.py — Video Diffusion Transformer (DiT) core.

Features
  * 3D rotary position embeddings (RoPE) over (time, height, width) token grids
  * FlashAttention-2 (flash_attn package) with automatic fallback to PyTorch SDPA
    (SDPA's CUDA flash backend is also FA2-based on bf16/fp16)
  * AdaLN-single (PixArt-alpha style): ONE shared timestep MLP -> 6*D modulation,
    plus a small per-block learned table added to it (far fewer params than per-block AdaLN-Zero MLPs)
  * QK-RMSNorm for bf16 stability at scale
  * Cross-attention to text tokens with padding mask
  * Gradient checkpointing, fp32-safe norms, ZeRO-3 friendly module boundaries

Shapes
  latent   x   : [B, C, T, H, W]   (from CausalVideoVAE, already scaled)
  timestep t   : [B] in [0, 1]     (rectified-flow time; 0 = data, 1 = noise)
  text     ctx : [B, L, text_dim]  ctx_mask: [B, L] bool (True = keep)
  output       : [B, C, T, H, W]   predicted velocity  v = noise - data
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:  # FlashAttention-2
    from flash_attn import flash_attn_func

    HAS_FLASH = True
except Exception:  # pragma: no cover
    flash_attn_func = None
    HAS_FLASH = False

__all__ = [
    "DiTConfig", "VideoDiT", "DiTBlock", "RoPE3D", "apply_rope",
    "patchify", "unpatchify", "AdaLNSingle", "HAS_FLASH",
]


# --------------------------------------------------------------------------- #
# Norms / small utils
# --------------------------------------------------------------------------- #
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * self.weight.float()).to(dtype)


class FP32LayerNorm(nn.LayerNorm):
    """LayerNorm computed in fp32 regardless of activation dtype (safe for pure-bf16 weights)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.float() if self.weight is not None else None
        b = self.bias.float() if self.bias is not None else None
        return F.layer_norm(x.float(), self.normalized_shape, w, b, self.eps).to(x.dtype)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale[:, None]) + shift[:, None]


# --------------------------------------------------------------------------- #
# 3D RoPE
# --------------------------------------------------------------------------- #
class RoPE3D(nn.Module):
    """Axial rotary embedding: head_dim is split into (t, h, w) sub-spaces, each rotated by its own
    coordinate. Pair i is (x[i], x[i + d/2]) ("rotate-half" layout), so cos/sin have width d/2."""

    def __init__(self, head_dim: int, theta: float = 10000.0, axes_dims: Optional[Tuple[int, int, int]] = None):
        super().__init__()
        if axes_dims is None:
            dh = (head_dim // 3) // 2 * 2
            dw = dh
            dt = head_dim - dh - dw
            axes_dims = (dt, dh, dw)
        assert sum(axes_dims) == head_dim and all(d % 2 == 0 and d > 0 for d in axes_dims), axes_dims
        self.axes_dims = axes_dims
        for name, d in zip(("t", "h", "w"), axes_dims):
            inv = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
            self.register_buffer(f"inv_{name}", inv, persistent=False)

    @torch.no_grad()
    def forward(self, grid: Tuple[int, int, int], device, t_offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        T, H, W = grid
        t = torch.arange(T, device=device, dtype=torch.float32) + t_offset
        h = torch.arange(H, device=device, dtype=torch.float32)
        w = torch.arange(W, device=device, dtype=torch.float32)
        ft = torch.outer(t, self.inv_t.to(device))[:, None, None, :].expand(T, H, W, -1)
        fh = torch.outer(h, self.inv_h.to(device))[None, :, None, :].expand(T, H, W, -1)
        fw = torch.outer(w, self.inv_w.to(device))[None, None, :, :].expand(T, H, W, -1)
        freqs = torch.cat([ft, fh, fw], dim=-1).reshape(T * H * W, -1)  # [N, d/2]
        return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [B, N, H, D]; cos/sin: [N, D/2] (fp32). Rotation done in fp32, result cast back."""
    d2 = x.shape[-1] // 2
    xf = x.float()
    x1, x2 = xf[..., :d2], xf[..., d2:]
    c, s = cos[None, :, None, :], sin[None, :, None, :]
    out = torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)
    return out.to(x.dtype)


# --------------------------------------------------------------------------- #
# Attention
# --------------------------------------------------------------------------- #
def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
    """q: [B, Nq, H, D], k/v: [B, Nk, H, D] -> [B, Nq, H, D].
    FlashAttention-2 when available, unmasked, on CUDA fp16/bf16; otherwise PyTorch SDPA."""
    if HAS_FLASH and attn_mask is None and q.is_cuda and q.dtype in (torch.float16, torch.bfloat16):
        return flash_attn_func(q, k, v)
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))
    o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    return o.transpose(1, 2)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, qk_norm: bool = True):
        super().__init__()
        assert dim % heads == 0
        self.heads, self.head_dim = heads, dim // heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, rope: Optional[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        B, N, C = x.shape
        q, k, v = self.qkv(x).view(B, N, 3, self.heads, self.head_dim).unbind(2)
        q, k = self.q_norm(q), self.k_norm(k)
        if rope is not None:
            q, k = apply_rope(q, *rope), apply_rope(k, *rope)
        o = attention(q, k, v)
        return self.proj(o.reshape(B, N, C))


class CrossAttention(nn.Module):
    def __init__(self, dim: int, heads: int, qk_norm: bool = True):
        super().__init__()
        self.heads, self.head_dim = heads, dim // heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor, ctx_mask: Optional[torch.Tensor]) -> torch.Tensor:
        B, N, C = x.shape
        M = ctx.shape[1]
        q = self.q_norm(self.q(x).view(B, N, self.heads, self.head_dim))
        k, v = self.kv(ctx).view(B, M, 2, self.heads, self.head_dim).unbind(2)
        k = self.k_norm(k)
        mask = ctx_mask[:, None, None, :] if ctx_mask is not None else None  # bool, True = attend
        o = attention(q, k, v, attn_mask=mask)
        return self.proj(o.reshape(B, N, C))


# --------------------------------------------------------------------------- #
# MLPs
# --------------------------------------------------------------------------- #
class GeluMLP(nn.Module):
    def __init__(self, dim: int, ratio: float):
        super().__init__()
        hid = int(dim * ratio)
        self.fc1, self.fc2 = nn.Linear(dim, hid), nn.Linear(hid, dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class SwiGLUMLP(nn.Module):
    def __init__(self, dim: int, ratio: float):
        super().__init__()
        hid = int(2 * dim * ratio / 3)
        hid = (hid + 63) // 64 * 64
        self.w12 = nn.Linear(dim, hid * 2, bias=False)
        self.w3 = nn.Linear(hid, dim, bias=False)

    def forward(self, x):
        a, b = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(a) * b)


# --------------------------------------------------------------------------- #
# Timestep embedding + AdaLN-single
# --------------------------------------------------------------------------- #
def sinusoidal_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class AdaLNSingle(nn.Module):
    """Shared across all blocks. Returns (modulation [B, 6D], embedded_t [B, D])."""

    def __init__(self, dim: int, freq_dim: int = 256, time_scale: float = 1000.0, n_mod: int = 6):
        super().__init__()
        self.freq_dim, self.time_scale = freq_dim, time_scale
        self.mlp = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.linear = nn.Linear(dim, n_mod * dim)

    def forward(self, t: torch.Tensor):
        emb = sinusoidal_embedding(t * self.time_scale, self.freq_dim).to(self.mlp[0].weight.dtype)
        e = self.mlp(emb)
        return self.linear(F.silu(e)), e


# --------------------------------------------------------------------------- #
# DiT block
# --------------------------------------------------------------------------- #
class DiTBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, qk_norm: bool = True, mlp_type: str = "gelu"):
        super().__init__()
        self.norm1 = FP32LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = SelfAttention(dim, heads, qk_norm)
        self.norm_cross = FP32LayerNorm(dim, eps=1e-6)
        self.cross = CrossAttention(dim, heads, qk_norm)
        self.norm2 = FP32LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = SwiGLUMLP(dim, mlp_ratio) if mlp_type == "swiglu" else GeluMLP(dim, mlp_ratio)
        # per-block table added to the shared modulation: (shift, scale, gate) x (attn, mlp)
        self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim**0.5)

    def forward(self, x, t_mod, ctx, ctx_mask, rope):
        B = x.shape[0]
        mod = self.scale_shift_table[None].to(t_mod.dtype) + t_mod.reshape(B, 6, -1)
        sh_a, sc_a, g_a, sh_m, sc_m, g_m = mod.unbind(1)
        x = x + g_a[:, None] * self.attn(modulate(self.norm1(x), sh_a, sc_a), rope)
        x = x + self.cross(self.norm_cross(x), ctx, ctx_mask)
        x = x + g_m[:, None] * self.mlp(modulate(self.norm2(x), sh_m, sc_m))
        return x


# --------------------------------------------------------------------------- #
# Patchify / unpatchify
# --------------------------------------------------------------------------- #
def patchify(x: torch.Tensor, patch: Tuple[int, int, int]) -> torch.Tensor:
    """[B, C, T, H, W] -> [B, N, pt*ph*pw*C] with token order (T', H', W')."""
    B, C, T, H, W = x.shape
    pt, ph, pw = patch
    x = x.view(B, C, T // pt, pt, H // ph, ph, W // pw, pw)
    x = x.permute(0, 2, 4, 6, 3, 5, 7, 1)
    return x.reshape(B, -1, pt * ph * pw * C)


def unpatchify(x: torch.Tensor, grid: Tuple[int, int, int], patch: Tuple[int, int, int], out_ch: int) -> torch.Tensor:
    """[B, N, pt*ph*pw*C] -> [B, C, T, H, W]."""
    B = x.shape[0]
    T, H, W = grid
    pt, ph, pw = patch
    x = x.view(B, T, H, W, pt, ph, pw, out_ch)
    x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)
    return x.reshape(B, out_ch, T * pt, H * ph, W * pw)


class PatchEmbed3D(nn.Module):
    def __init__(self, in_ch: int, dim: int, patch: Tuple[int, int, int]):
        super().__init__()
        self.patch = patch
        self.proj = nn.Conv3d(in_ch, dim, kernel_size=patch, stride=patch)

    def forward(self, x: torch.Tensor):
        assert all(s % p == 0 for s, p in zip(x.shape[2:], self.patch)), (x.shape, self.patch)
        x = self.proj(x)
        grid = tuple(x.shape[2:])
        return x.flatten(2).transpose(1, 2), grid


class CaptionProjector(nn.Module):
    def __init__(self, text_dim: int, dim: int):
        super().__init__()
        self.norm = RMSNorm(text_dim)
        self.fc1 = nn.Linear(text_dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(self.norm(c)), approximate="tanh"))


class FinalLayer(nn.Module):
    def __init__(self, dim: int, patch: Tuple[int, int, int], out_ch: int):
        super().__init__()
        self.norm = FP32LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, patch[0] * patch[1] * patch[2] * out_ch)
        self.scale_shift_table = nn.Parameter(torch.randn(2, dim) / dim**0.5)

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        shift, scale = (self.scale_shift_table[None].to(e.dtype) + e[:, None]).unbind(1)
        return self.linear(modulate(self.norm(x), shift, scale))


# --------------------------------------------------------------------------- #
# Full model
# --------------------------------------------------------------------------- #
@dataclass
class DiTConfig:
    in_channels: int = 16
    hidden_size: int = 1152
    depth: int = 28
    num_heads: int = 16
    patch_size: Tuple[int, int, int] = (1, 2, 2)
    text_dim: int = 4096
    mlp_ratio: float = 4.0
    mlp_type: str = "gelu"        # "gelu" | "swiglu"
    qk_norm: bool = True
    rope_theta: float = 10000.0
    gradient_checkpointing: bool = False
    checkpoint_every: int = 1      # checkpoint every n-th block (1 = all blocks)


class VideoDiT(nn.Module):
    _no_split_modules = ["DiTBlock"]  # FSDP / ZeRO-3 wrapping unit

    def __init__(self, cfg: DiTConfig | None = None):
        super().__init__()
        self.cfg = cfg = cfg or DiTConfig()
        D = cfg.hidden_size
        self.patch_embed = PatchEmbed3D(cfg.in_channels, D, cfg.patch_size)
        self.rope = RoPE3D(D // cfg.num_heads, cfg.rope_theta)
        self.adaln = AdaLNSingle(D)
        self.caption = CaptionProjector(cfg.text_dim, D)
        self.blocks = nn.ModuleList(
            [DiTBlock(D, cfg.num_heads, cfg.mlp_ratio, cfg.qk_norm, cfg.mlp_type) for _ in range(cfg.depth)]
        )
        self.final = FinalLayer(D, cfg.patch_size, cfg.in_channels)
        self._init_weights()

    def _init_weights(self):
        def basic(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(basic)
        w = self.patch_embed.proj.weight
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        nn.init.zeros_(self.patch_embed.proj.bias)
        for m in self.adaln.mlp:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
        for blk in self.blocks:  # residual branches start as identity
            nn.init.zeros_(blk.cross.proj.weight)
            nn.init.zeros_(blk.cross.proj.bias)
        nn.init.zeros_(self.final.linear.weight)
        nn.init.zeros_(self.final.linear.bias)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        ctx: torch.Tensor,
        ctx_mask: Optional[torch.Tensor] = None,
        t_offset: int = 0,
    ) -> torch.Tensor:
        tokens, grid = self.patch_embed(x)
        cos, sin = self.rope(grid, x.device, t_offset)
        t_mod, e = self.adaln(t)
        ctx = self.caption(ctx.to(tokens.dtype))
        if ctx_mask is not None:
            ctx_mask = ctx_mask.bool()
            empty = ~ctx_mask.any(dim=1)  # avoid NaN rows if a prompt is fully masked
            if empty.any():
                ctx_mask = ctx_mask.clone()
                ctx_mask[empty, 0] = True

        use_ckpt = self.cfg.gradient_checkpointing and self.training
        for i, blk in enumerate(self.blocks):
            if use_ckpt and i % self.cfg.checkpoint_every == 0:
                tokens = checkpoint(blk, tokens, t_mod, ctx, ctx_mask, (cos, sin), use_reentrant=False)
            else:
                tokens = blk(tokens, t_mod, ctx, ctx_mask, (cos, sin))
        out = self.final(tokens, e)
        return unpatchify(out, grid, self.cfg.patch_size, self.cfg.in_channels)


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def _selftest():
    torch.manual_seed(0)

    # 1) RoPE: q.k depends only on relative (t, h, w) offset
    rope = RoPE3D(64)
    T, H, W = 4, 6, 6
    cos, sin = rope((T, H, W), "cpu")
    q = torch.randn(1, 1, 1, 64).expand(1, T * H * W, 1, 64).contiguous()
    k = torch.randn(1, 1, 1, 64).expand(1, T * H * W, 1, 64).contiguous()
    qr, kr = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    idx = lambda t, h, w: t * H * W + h * W + w
    a = (qr[0, idx(0, 0, 0), 0] * kr[0, idx(1, 2, 3), 0]).sum()
    b = (qr[0, idx(1, 1, 1), 0] * kr[0, idx(2, 3, 4), 0]).sum()
    assert torch.allclose(a, b, atol=1e-4), (a, b)
    print("RoPE relative-offset invariance OK")

    # 2) patchify / unpatchify round trip
    x = torch.randn(2, 16, 4, 8, 8)
    p = (2, 2, 2)
    assert torch.equal(unpatchify(patchify(x, p), (2, 4, 4), p, 16), x)
    print("patchify round trip OK")

    # 3) forward / backward / mask invariance
    cfg = DiTConfig(in_channels=16, hidden_size=128, depth=2, num_heads=4, text_dim=64, gradient_checkpointing=True)
    model = VideoDiT(cfg).train()
    nn.init.normal_(model.final.linear.weight, std=0.02)
    for blk in model.blocks:
        nn.init.normal_(blk.cross.proj.weight, std=0.02)
    x = torch.randn(2, 16, 3, 8, 8)
    t = torch.rand(2)
    ctx = torch.randn(2, 10, 64)
    mask = torch.ones(2, 10, dtype=torch.bool)
    mask[1, 5:] = False
    out = model(x, t, ctx, mask)
    assert out.shape == x.shape
    out.pow(2).mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
    print(f"forward/backward OK ({model.num_parameters/1e6:.2f}M params, flash_attn={HAS_FLASH})")

    model.eval()
    ctx2 = ctx.clone()
    ctx2[1, 5:] = torch.randn(5, 64)
    with torch.no_grad():
        o1, o2 = model(x, t, ctx, mask), model(x, t, ctx2, mask)
    assert torch.allclose(o1, o2, atol=1e-5), "masked text tokens leaked into output"
    print("text padding mask OK")


if __name__ == "__main__":
    _selftest()
