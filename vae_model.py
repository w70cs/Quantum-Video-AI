"""
vae_model.py — 3D Causal Video VAE.

Compression: 4x temporal, 8x spatial, latent channels configurable (default 16).
Causality: frame t never sees frames > t. The first frame is encoded independently,
so T = 1 + 4k input frames map to T' = 1 + k latent frames (image/video joint training).

Usage:
    vae = CausalVideoVAE()
    posterior = vae.encode(video)          # video: [B, 3, T, H, W] in [-1, 1], T = 1 + 4k
    z = posterior.sample()                 # [B, C_lat, 1 + k, H/8, W/8]
    recon = vae.decode(z)                  # [B, 3, T, H, W]
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

IntOrTuple = Union[int, Tuple[int, int, int]]


def _triple(x: IntOrTuple) -> Tuple[int, int, int]:
    return (x, x, x) if isinstance(x, int) else tuple(x)


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
class CausalConv3d(nn.Module):
    """3D conv, causal in time (front replicate-padding with the first frame),
    zero-padded in space. Optional asymmetric spatial padding for stride-2 downsampling."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: IntOrTuple = 3,
        stride: IntOrTuple = 1,
        spatial_pad: Union[str, Tuple[int, int, int, int]] = "same",
    ):
        super().__init__()
        kt, kh, kw = _triple(kernel_size)
        self.time_pad = kt - 1
        if spatial_pad == "same":
            self.spatial_pad = (kw // 2, kw // 2, kh // 2, kh // 2)  # (W_l, W_r, H_t, H_b)
        else:
            self.spatial_pad = tuple(spatial_pad)
        self.conv = nn.Conv3d(in_ch, out_ch, (kt, kh, kw), stride=_triple(stride))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.time_pad > 0:
            first = x[:, :, :1].expand(-1, -1, self.time_pad, -1, -1)
            x = torch.cat([first, x], dim=2)
        if any(self.spatial_pad):
            x = F.pad(x, self.spatial_pad)
        return self.conv(x)


class RMSNorm3d(nn.Module):
    """Channel-wise RMSNorm for [B, C, T, H, W] (stable in bf16, no batch/time stats)."""

    def __init__(self, ch: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, ch, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(dim=1, keepdim=True) + self.eps)
        return (xf.to(dtype)) * self.weight.to(dtype)


class ResBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = RMSNorm3d(in_ch)
        self.conv1 = CausalConv3d(in_ch, out_ch, 3)
        self.norm2 = RMSNorm3d(out_ch)
        self.drop = nn.Dropout(dropout)
        self.conv2 = CausalConv3d(out_ch, out_ch, 3)
        self.skip = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(self.drop(F.silu(self.norm2(h))))
        return self.skip(x) + h


class SpatialAttnBlock(nn.Module):
    """Per-frame spatial self-attention (used in the mid block only; cheap and causal-safe)."""

    def __init__(self, ch: int, head_dim: int = 64):
        super().__init__()
        self.heads = max(1, ch // head_dim)
        self.norm = RMSNorm3d(ch)
        self.qkv = nn.Conv3d(ch, ch * 3, 1)
        self.proj = nn.Conv3d(ch, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape
        qkv = self.qkv(self.norm(x))  # [B, 3C, T, H, W]
        qkv = qkv.permute(0, 2, 3, 4, 1).reshape(B * T, H * W, 3, self.heads, C // self.heads)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # [BT, heads, HW, d]
        o = F.scaled_dot_product_attention(q, k, v)
        o = o.transpose(1, 2).reshape(B, T, H, W, C).permute(0, 4, 1, 2, 3)
        return x + self.proj(o)


class SpatialDown(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        # asymmetric pad (right/bottom) + stride-2 3x3 -> exact /2 for even sizes
        self.conv = CausalConv3d(ch, ch, (1, 3, 3), stride=(1, 2, 2), spatial_pad=(0, 1, 0, 1))

    def forward(self, x):
        return self.conv(x)


class TemporalDown(nn.Module):
    """Causal stride-2 temporal conv. T = 1 + 2k  ->  1 + k."""

    def __init__(self, ch: int):
        super().__init__()
        self.conv = CausalConv3d(ch, ch, (3, 1, 1), stride=(2, 1, 1), spatial_pad=(0, 0, 0, 0))

    def forward(self, x):
        return self.conv(x)


class SpatialUp(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = CausalConv3d(ch, ch, 3)

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = F.interpolate(x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W), scale_factor=2.0, mode="nearest")
        x = x.reshape(B, T, C, H * 2, W * 2).permute(0, 2, 1, 3, 4)
        return self.conv(x)


class TemporalUp(nn.Module):
    """Causal temporal 2x upsample that keeps the first frame intact: T = 1 + k -> 1 + 2k."""

    def __init__(self, ch: int):
        super().__init__()
        self.conv = CausalConv3d(ch, ch, (3, 1, 1), spatial_pad=(0, 0, 0, 0))

    def forward(self, x):
        if x.shape[2] > 1:
            first, rest = x[:, :, :1], x[:, :, 1:]
            rest = rest.repeat_interleave(2, dim=2)
            x = torch.cat([first, rest], dim=2)
        return self.conv(x)


# --------------------------------------------------------------------------- #
# Distribution
# --------------------------------------------------------------------------- #
class DiagonalGaussian:
    def __init__(self, params: torch.Tensor):
        self.mean, logvar = params.float().chunk(2, dim=1)
        self.logvar = torch.clamp(logvar, -30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)

    def sample(self, generator: torch.Generator | None = None) -> torch.Tensor:
        eps = torch.randn(self.mean.shape, device=self.mean.device, dtype=self.mean.dtype, generator=generator)
        return self.mean + self.std * eps

    def mode(self) -> torch.Tensor:
        return self.mean

    def kl(self) -> torch.Tensor:
        return 0.5 * torch.sum(self.mean.pow(2) + self.logvar.exp() - 1.0 - self.logvar, dim=[1, 2, 3, 4])


# --------------------------------------------------------------------------- #
# Encoder / Decoder
# --------------------------------------------------------------------------- #
@dataclass
class VAEConfig:
    in_channels: int = 3
    latent_channels: int = 16
    base_channels: int = 128
    channel_mult: Tuple[int, ...] = (1, 2, 4, 4)       # 4 stages -> 3 downsamples -> 8x spatial
    temporal_down: Tuple[bool, ...] = (False, True, True)  # per downsample: 2 temporal halvings -> 4x
    num_res_blocks: int = 2
    dropout: float = 0.0
    scaling_factor: float = 1.0   # set from dataset latent std before DiT training
    shift_factor: float = 0.0
    gradient_checkpointing: bool = False


class Encoder3D(nn.Module):
    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.cfg = cfg
        ch = cfg.base_channels
        self.conv_in = CausalConv3d(cfg.in_channels, ch, 3)
        self.stages = nn.ModuleList()
        in_ch = ch
        for i, m in enumerate(cfg.channel_mult):
            out_ch = cfg.base_channels * m
            blocks = nn.ModuleList()
            for _ in range(cfg.num_res_blocks):
                blocks.append(ResBlock3D(in_ch, out_ch, cfg.dropout))
                in_ch = out_ch
            down = nn.ModuleList()
            if i < len(cfg.channel_mult) - 1:
                down.append(SpatialDown(in_ch))
                if cfg.temporal_down[i]:
                    down.append(TemporalDown(in_ch))
            self.stages.append(nn.ModuleDict({"blocks": blocks, "down": down}))
        self.mid1 = ResBlock3D(in_ch, in_ch, cfg.dropout)
        self.mid_attn = SpatialAttnBlock(in_ch)
        self.mid2 = ResBlock3D(in_ch, in_ch, cfg.dropout)
        self.norm_out = RMSNorm3d(in_ch)
        self.conv_out = CausalConv3d(in_ch, 2 * cfg.latent_channels, 3)

    def _run(self, mod, x):
        if self.cfg.gradient_checkpointing and self.training and x.requires_grad:
            return checkpoint(mod, x, use_reentrant=False)
        return mod(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        for stage in self.stages:
            for blk in stage["blocks"]:
                x = self._run(blk, x)
            for d in stage["down"]:
                x = d(x)
        x = self._run(self.mid1, x)
        x = self._run(self.mid_attn, x)
        x = self._run(self.mid2, x)
        return self.conv_out(F.silu(self.norm_out(x)))


class Decoder3D(nn.Module):
    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.cfg = cfg
        mult = list(cfg.channel_mult)
        ch = cfg.base_channels * mult[-1]
        self.conv_in = CausalConv3d(cfg.latent_channels, ch, 3)
        self.mid1 = ResBlock3D(ch, ch, cfg.dropout)
        self.mid_attn = SpatialAttnBlock(ch)
        self.mid2 = ResBlock3D(ch, ch, cfg.dropout)
        self.stages = nn.ModuleList()
        in_ch = ch
        n = len(mult)
        for i, m in enumerate(reversed(mult)):
            out_ch = cfg.base_channels * m
            blocks = nn.ModuleList()
            for _ in range(cfg.num_res_blocks + 1):
                blocks.append(ResBlock3D(in_ch, out_ch, cfg.dropout))
                in_ch = out_ch
            up = nn.ModuleList()
            if i < n - 1:
                # mirror the encoder: the last encoder downsample is the first decoder upsample
                enc_idx = n - 2 - i
                up.append(SpatialUp(in_ch))
                if cfg.temporal_down[enc_idx]:
                    up.append(TemporalUp(in_ch))
            self.stages.append(nn.ModuleDict({"blocks": blocks, "up": up}))
        self.norm_out = RMSNorm3d(in_ch)
        self.conv_out = CausalConv3d(in_ch, cfg.in_channels, 3)

    def _run(self, mod, x):
        if self.cfg.gradient_checkpointing and self.training and x.requires_grad:
            return checkpoint(mod, x, use_reentrant=False)
        return mod(x)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(z)
        x = self._run(self.mid1, x)
        x = self._run(self.mid_attn, x)
        x = self._run(self.mid2, x)
        for stage in self.stages:
            for blk in stage["blocks"]:
                x = self._run(blk, x)
            for u in stage["up"]:
                x = u(x)
        return self.conv_out(F.silu(self.norm_out(x)))


# --------------------------------------------------------------------------- #
# Full VAE
# --------------------------------------------------------------------------- #
class CausalVideoVAE(nn.Module):
    def __init__(self, cfg: VAEConfig | None = None):
        super().__init__()
        self.cfg = cfg or VAEConfig()
        self.encoder = Encoder3D(self.cfg)
        self.decoder = Decoder3D(self.cfg)
        self.quant_conv = nn.Conv3d(2 * self.cfg.latent_channels, 2 * self.cfg.latent_channels, 1)
        self.post_quant_conv = nn.Conv3d(self.cfg.latent_channels, self.cfg.latent_channels, 1)
        self.t_ratio = 2 ** sum(self.cfg.temporal_down)
        self.s_ratio = 2 ** (len(self.cfg.channel_mult) - 1)

    # ---- helpers -------------------------------------------------------- #
    def latent_shape(self, T: int, H: int, W: int) -> Tuple[int, int, int]:
        assert (T - 1) % self.t_ratio == 0, f"T must be 1 + {self.t_ratio}k, got {T}"
        assert H % self.s_ratio == 0 and W % self.s_ratio == 0
        return (T - 1) // self.t_ratio + 1, H // self.s_ratio, W // self.s_ratio

    def normalize_latent(self, z: torch.Tensor) -> torch.Tensor:
        return (z - self.cfg.shift_factor) * self.cfg.scaling_factor

    def denormalize_latent(self, z: torch.Tensor) -> torch.Tensor:
        return z / self.cfg.scaling_factor + self.cfg.shift_factor

    # ---- API ------------------------------------------------------------ #
    def encode(self, x: torch.Tensor) -> DiagonalGaussian:
        self.latent_shape(*x.shape[2:])
        return DiagonalGaussian(self.quant_conv(self.encoder(x)))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.post_quant_conv(z))

    def forward(self, x: torch.Tensor, sample_posterior: bool = True):
        post = self.encode(x)
        z = post.sample() if sample_posterior else post.mode()
        recon = self.decode(z.to(x.dtype))
        return recon, post

    # ---- memory-bounded spatial tiling for high-res inference ------------ #
    @torch.no_grad()
    def tiled_decode(self, z: torch.Tensor, tile: int = 32, overlap: int = 8) -> torch.Tensor:
        """Spatially tiled decode with linear blending. tile/overlap are in latent pixels."""
        B, C, T, H, W = z.shape
        if H <= tile and W <= tile:
            return self.decode(z)
        s = self.s_ratio
        stride = tile - overlap
        out_T = (T - 1) * self.t_ratio + 1
        out = torch.zeros(B, self.cfg.in_channels, out_T, H * s, W * s, device=z.device, dtype=torch.float32)
        weight = torch.zeros(1, 1, 1, H * s, W * s, device=z.device, dtype=torch.float32)
        ys = list(range(0, max(H - overlap, 1), stride))
        xs = list(range(0, max(W - overlap, 1), stride))
        for y in ys:
            for x in xs:
                y1, x1 = min(y + tile, H), min(x + tile, W)
                y0, x0 = max(y1 - tile, 0), max(x1 - tile, 0)
                patch = self.decode(z[..., y0:y1, x0:x1]).float()
                ph, pw = patch.shape[-2:]
                wy = torch.ones(ph, device=z.device)
                wx = torch.ones(pw, device=z.device)
                ov = overlap * s
                ramp = torch.linspace(0, 1, ov + 2, device=z.device)[1:-1]
                if y0 > 0:
                    wy[:ov] = ramp
                if y1 < H:
                    wy[-ov:] = ramp.flip(0)
                if x0 > 0:
                    wx[:ov] = ramp
                if x1 < W:
                    wx[-ov:] = ramp.flip(0)
                w = (wy[:, None] * wx[None, :])[None, None, None]
                out[..., y0 * s : y1 * s, x0 * s : x1 * s] += patch * w
                weight[..., y0 * s : y1 * s, x0 * s : x1 * s] += w
        return (out / weight.clamp_min(1e-6)).to(z.dtype)


# --------------------------------------------------------------------------- #
# Loss (reconstruction + KL; plug LPIPS / GAN discriminator in train_vae stage)
# --------------------------------------------------------------------------- #
def vae_loss(recon: torch.Tensor, target: torch.Tensor, post: DiagonalGaussian, kl_weight: float = 1e-6,
             lpips_fn=None, lpips_weight: float = 0.1):
    rec = F.l1_loss(recon.float(), target.float())
    kl = post.kl().mean() / target[0].numel()  # per-sample KL sum, normalised like the mean-L1 term
    total = rec + kl_weight * kl
    logs = {"rec_l1": rec.detach(), "kl": kl.detach()}
    if lpips_fn is not None:
        B, C, T, H, W = recon.shape
        idx = torch.randint(0, T, (min(T, 4),), device=recon.device)  # subsample frames for speed
        r = recon[:, :, idx].permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
        t = target[:, :, idx].permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
        lp = lpips_fn(r.float(), t.float()).mean()
        total = total + lpips_weight * lp
        logs["lpips"] = lp.detach()
    return total, logs


if __name__ == "__main__":
    cfg = VAEConfig(base_channels=32, channel_mult=(1, 2, 2, 4), num_res_blocks=1)
    vae = CausalVideoVAE(cfg).eval()
    x = torch.randn(1, 3, 9, 64, 64)
    with torch.no_grad():
        recon, post = vae(x)
        z = post.mode()
    print("input", tuple(x.shape), "latent", tuple(z.shape), "recon", tuple(recon.shape))
    assert recon.shape == x.shape and z.shape[2:] == vae.latent_shape(9, 64, 64)
    # causality check: changing a late frame must not alter earlier latents
    x2 = x.clone()
    x2[:, :, -1] += 1.0
    with torch.no_grad():
        z2 = vae.encode(x2).mode()
    assert torch.allclose(z[:, :, :-1], z2[:, :, :-1], atol=1e-5), "causality violated"
    print("causality OK")
