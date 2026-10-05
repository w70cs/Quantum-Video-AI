"""
pipeline.py — inference / sampling for the video DiT (rectified flow).

Conventions (must match train.py)
  x_t = (1 - t) * x0 + t * noise,  t in [0, 1]  (t=1 pure noise, t=0 data)
  the DiT predicts the velocity  v = noise - x0 = dx/dt
  sampling integrates dx/dt from t=1 down to t=0 on a shifted time grid.

Solvers (NFE = network evaluations per step)
  euler : 1 NFE, first order
  ab2   : 1 NFE, second-order Adams-Bashforth multistep (reuses the previous velocity)
  heun  : 2 NFE (1 on the last step), second-order predictor-corrector

Checkpoint directory layout (written by train.py, read here)
  <dir>/vae/{config.json, model.safetensors|model.pt}
  <dir>/dit/{config.json, model.safetensors|model.pt}
  <dir>/pipeline.json        {"text_encoder": {...TextEncoderConfig...}}

CLI
  python pipeline.py --ckpt export/ --prompt "a corgi surfing" --out out.mp4 --frames 121 --height 704 --width 1280
  python pipeline.py --selftest
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import tempfile
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Union

import torch
import torch.nn as nn

from dit_blocks import DiTConfig, VideoDiT
from text_encoder import BaseTextEncoder, TextEncoderConfig, build_text_encoder
from vae_model import CausalVideoVAE, VAEConfig

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

__all__ = [
    "FlowMatchScheduler", "VideoPipeline", "PipelineOutput", "sample_flow", "shift_time", "auto_shift",
    "save_component", "load_component", "config_from_dict", "write_pipeline_index", "read_pipeline_index",
    "save_video",
]


# --------------------------------------------------------------------------- #
# Time shifting + scheduler
# --------------------------------------------------------------------------- #
def shift_time(t, shift: float):
    """Monotone map of [0,1] onto itself that spends more steps at high noise when shift > 1.
    Works on floats and tensors. Used for BOTH training-time sampling and inference grids."""
    return shift * t / (1.0 + (shift - 1.0) * t)


def auto_shift(num_tokens: int, base_tokens: int = 4096, base_shift: float = 3.0,
               lo: float = 1.0, hi: float = 8.0) -> float:
    """Heuristic: shift grows ~ sqrt(sequence length) (SD3 observation). Tune per model."""
    return float(min(max(base_shift * math.sqrt(num_tokens / base_tokens), lo), hi))


class FlowMatchScheduler:
    def __init__(self, shift: float = 5.0, solver: str = "euler"):
        assert solver in ("euler", "heun", "ab2"), solver
        self.shift, self.solver = shift, solver

    def timesteps(self, num_steps: int) -> List[float]:
        """num_steps + 1 monotone-decreasing floats, first 1.0, last 0.0."""
        grid = [1.0 - i / num_steps for i in range(num_steps + 1)]
        return [float(shift_time(g, self.shift)) for g in grid]


@torch.no_grad()
def sample_flow(
    velocity_fn: Callable[[torch.Tensor, float], torch.Tensor],
    x: torch.Tensor,
    timesteps: Sequence[float],
    solver: str = "euler",
    progress: bool = False,
    callback: Optional[Callable[[int, float, torch.Tensor], None]] = None,
) -> torch.Tensor:
    """Integrate dx/dt = velocity_fn(x, t) from timesteps[0] (=1, noise) to timesteps[-1] (=0, data)."""
    n = len(timesteps) - 1
    it = range(n)
    if progress and tqdm is not None:
        it = tqdm(it, total=n, desc="denoise")
    prev_v, prev_dt = None, None
    for i in it:
        t, t_next = timesteps[i], timesteps[i + 1]
        dt = t_next - t  # negative
        v = velocity_fn(x, t)
        if solver == "euler":
            x = x + dt * v
        elif solver == "ab2":
            if prev_v is None:
                x = x + dt * v
            else:
                r = dt / prev_dt
                x = x + dt * (v + 0.5 * r * (v - prev_v))
            prev_v, prev_dt = v, dt
        elif solver == "heun":
            x_pred = x + dt * v
            if i < n - 1:
                v2 = velocity_fn(x_pred, t_next)
                x = x + dt * 0.5 * (v + v2)
            else:
                x = x_pred
        else:
            raise ValueError(solver)
        if callback is not None:
            callback(i, t_next, x)
    return x


# --------------------------------------------------------------------------- #
# Checkpoint IO
# --------------------------------------------------------------------------- #
def config_from_dict(cls, d: dict):
    """Rebuild a dataclass config from JSON (lists -> tuples, unknown keys ignored)."""
    names = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items() if k in names})


def save_component(module: Optional[nn.Module], cfg, out_dir: str, state_dict: Optional[dict] = None,
                   dtype: Optional[torch.dtype] = None) -> None:
    out_dir = str(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=2)
    sd = state_dict if state_dict is not None else module.state_dict()
    out = {}
    for k, v in sd.items():
        v = v.detach()
        out[k] = (v.to(device="cpu", dtype=dtype) if (dtype is not None and v.is_floating_point())
                  else v.to("cpu")).contiguous()
    try:
        from safetensors.torch import save_file

        save_file(out, os.path.join(out_dir, "model.safetensors"))
    except ImportError:
        torch.save(out, os.path.join(out_dir, "model.pt"))


def load_component(model_cls, cfg_cls, path: str, device="cpu", dtype: Optional[torch.dtype] = None):
    path = str(path)
    with open(os.path.join(path, "config.json")) as f:
        cfg = config_from_dict(cfg_cls, json.load(f))
    prev = torch.get_default_dtype()
    if dtype is not None:
        torch.set_default_dtype(dtype)  # build parameters directly in the target dtype
    try:
        with torch.device(device):
            model = model_cls(cfg)
    finally:
        torch.set_default_dtype(prev)
    st_path = os.path.join(path, "model.safetensors")
    if os.path.exists(st_path):
        from safetensors.torch import load_file

        sd = load_file(st_path)
    else:
        sd = torch.load(os.path.join(path, "model.pt"), map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=True)
    return model.eval().requires_grad_(False)


def write_pipeline_index(path: str, text_cfg: TextEncoderConfig) -> None:
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "pipeline.json"), "w") as f:
        json.dump({"text_encoder": dataclasses.asdict(text_cfg)}, f, indent=2)


def read_pipeline_index(path: str) -> TextEncoderConfig:
    with open(os.path.join(path, "pipeline.json")) as f:
        return config_from_dict(TextEncoderConfig, json.load(f)["text_encoder"])


def save_video(frames: torch.Tensor, path: str, fps: int = 24) -> None:
    """frames: [T, H, W, 3] uint8."""
    arr = frames.cpu().numpy()
    try:
        import imageio.v2 as imageio

        imageio.mimwrite(path, list(arr), fps=fps, codec="libx264", quality=8, macro_block_size=1)
    except ImportError:
        from torchvision.io import write_video

        write_video(path, torch.from_numpy(arr), fps=fps)


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
@dataclass
class PipelineOutput:
    frames: torch.Tensor                       # [B, T, H, W, 3] uint8, CPU
    latents: Optional[torch.Tensor] = None     # [B, C, T', H', W'] normalised latents, CPU
    seed: int = 0

    def save(self, path: str, fps: int = 24, index: int = 0) -> None:
        save_video(self.frames[index], path, fps)


def _empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class VideoPipeline:
    def __init__(
        self,
        vae: CausalVideoVAE,
        dit: VideoDiT,
        text_encoder: BaseTextEncoder,
        device: Union[str, torch.device] = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        offload: bool = False,
    ):
        self.device, self.dtype, self.offload = torch.device(device), dtype, offload
        self.vae, self.dit, self.text_encoder = vae.eval(), dit.eval(), text_encoder
        if not offload:
            for m in (self.vae, self.dit, self.text_encoder):
                m.to(self.device)
        assert dit.cfg.in_channels == vae.cfg.latent_channels, "DiT/VAE latent channel mismatch"

    # ---- loading / saving ------------------------------------------------ #
    @classmethod
    def from_pretrained(cls, path: str, device=None, dtype=torch.bfloat16, vae_dtype=None,
                        offload: bool = False, text_encoder: Optional[BaseTextEncoder] = None):
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        build_dev = "cpu" if offload else device
        vae = load_component(CausalVideoVAE, VAEConfig, os.path.join(path, "vae"), build_dev, vae_dtype or dtype)
        dit = load_component(VideoDiT, DiTConfig, os.path.join(path, "dit"), build_dev, dtype)
        if text_encoder is None:
            text_encoder = build_text_encoder(read_pipeline_index(path))
        return cls(vae, dit, text_encoder, device, dtype, offload)

    def save_pretrained(self, path: str, text_cfg: TextEncoderConfig) -> None:
        save_component(self.vae, self.vae.cfg, os.path.join(path, "vae"))
        save_component(self.dit, self.dit.cfg, os.path.join(path, "dit"))
        write_pipeline_index(path, text_cfg)

    def compile_blocks(self) -> None:
        """Optional: torch.compile each DiT block (first call is slow, later calls faster)."""
        for blk in self.dit.blocks:
            blk.compile()

    # ---- helpers ---------------------------------------------------------- #
    def _autocast(self):
        on = self.device.type == "cuda" and self.dtype != torch.float32
        return torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=on)

    def _encode_text(self, prompts, negatives, do_cfg):
        te = self.text_encoder
        if self.offload:
            te.to(self.device)
        emb, mask = te(prompts)
        n_emb = n_mask = None
        if do_cfg:
            n_emb, n_mask = te(negatives)
        if self.offload:
            te.to("cpu")
            _empty_cache()
        return emb, mask, n_emb, n_mask

    # ---- main entry ------------------------------------------------------- #
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, Sequence[str]],
        negative_prompt: Union[str, Sequence[str]] = "",
        num_frames: int = 121,
        height: int = 704,
        width: int = 1280,
        num_steps: int = 30,
        guidance_scale: float = 5.0,
        guidance_rescale: float = 0.0,
        solver: str = "euler",
        shift: Optional[float] = None,
        seed: Optional[int] = None,
        tiled_decode: bool = True,
        decode_tile: int = 32,
        decode_overlap: int = 8,
        return_latents: bool = False,
        progress: bool = True,
        callback: Optional[Callable[[int, float, torch.Tensor], None]] = None,
    ) -> PipelineOutput:
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        B = len(prompts)
        negatives = [negative_prompt] * B if isinstance(negative_prompt, str) else list(negative_prompt)
        assert len(negatives) == B, "negative_prompt must be a string or match len(prompt)"
        do_cfg = guidance_scale > 1.0

        # ---- shape validation ----
        pt, ph, pw = self.dit.cfg.patch_size
        tr, sr = self.vae.t_ratio, self.vae.s_ratio
        if (num_frames - 1) % tr:
            raise ValueError(f"num_frames must be 1 + {tr}k, got {num_frames}")
        if height % (sr * ph) or width % (sr * pw):
            raise ValueError(f"height/width must be multiples of {sr * ph}/{sr * pw}, got {height}x{width}")
        Tl, Hl, Wl = self.vae.latent_shape(num_frames, height, width)
        if Tl % pt:
            raise ValueError(f"latent frames {Tl} not divisible by temporal patch {pt}")
        C = self.dit.cfg.in_channels
        num_tokens = (Tl // pt) * (Hl // ph) * (Wl // pw)
        if shift is None:
            shift = auto_shift(num_tokens)

        # ---- text ----
        emb, mask, n_emb, n_mask = self._encode_text(prompts, negatives, do_cfg)
        emb, mask = emb.to(self.device), mask.to(self.device)
        if do_cfg:
            ctx = torch.cat([n_emb.to(self.device), emb])
            ctx_mask = torch.cat([n_mask.to(self.device), mask])
        else:
            ctx, ctx_mask = emb, mask

        # ---- noise (per-sample seeds: results independent of batch composition) ----
        if seed is None:
            seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
        noises = []
        for i in range(B):
            g = torch.Generator(device=self.device).manual_seed(seed + i)
            noises.append(torch.randn(C, Tl, Hl, Wl, generator=g, device=self.device, dtype=torch.float32))
        x = torch.stack(noises)

        # ---- denoise ----
        if self.offload:
            self.dit.to(self.device)

        def velocity_fn(x_t: torch.Tensor, t: float) -> torch.Tensor:
            n = x_t.shape[0]
            xin = torch.cat([x_t, x_t]) if do_cfg else x_t
            tt = torch.full((xin.shape[0],), t, device=self.device, dtype=torch.float32)
            with self._autocast():
                out = self.dit(xin.to(self.dtype), tt, ctx, ctx_mask).float()
            if not do_cfg:
                return out
            v_u, v_c = out[:n], out[n:]
            v = v_u + guidance_scale * (v_c - v_u)
            if guidance_rescale > 0:
                dims = tuple(range(1, v.ndim))
                v_r = v * (v_c.std(dim=dims, keepdim=True) / (v.std(dim=dims, keepdim=True) + 1e-8))
                v = guidance_rescale * v_r + (1.0 - guidance_rescale) * v
            return v

        ts = FlowMatchScheduler(shift, solver).timesteps(num_steps)
        x = sample_flow(velocity_fn, x, ts, solver, progress, callback)
        if self.offload:
            self.dit.to("cpu")
            _empty_cache()
            self.vae.to(self.device)

        # ---- decode ----
        vae_dtype = next(self.vae.parameters()).dtype
        z = self.vae.denormalize_latent(x.to(vae_dtype))
        if tiled_decode:
            video = self.vae.tiled_decode(z, tile=decode_tile, overlap=decode_overlap)
        else:
            video = self.vae.decode(z)
        if self.offload:
            self.vae.to("cpu")
            _empty_cache()
        frames = ((video.float().clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8)
        frames = frames.permute(0, 2, 3, 4, 1).cpu()  # [B, T, H, W, 3]
        return PipelineOutput(frames, x.cpu() if return_latents else None, seed)


# --------------------------------------------------------------------------- #
# CLI + self-test
# --------------------------------------------------------------------------- #
def _selftest():
    torch.manual_seed(0)

    # 1) every solver reproduces the data exactly for the constant velocity of a straight path
    x0, noise = torch.randn(2, 4, 3, 8, 8), torch.randn(2, 4, 3, 8, 8)
    for solver in ("euler", "ab2", "heun"):
        for steps in (1, 3, 10):
            for shift in (1.0, 5.0):
                ts = FlowMatchScheduler(shift, solver).timesteps(steps)
                assert ts[0] == 1.0 and ts[-1] == 0.0 and all(a > b for a, b in zip(ts, ts[1:]))
                out = sample_flow(lambda x, t: noise - x0, noise.clone(), ts, solver)
                assert torch.allclose(out, x0, atol=1e-5), (solver, steps, shift)
    print("scheduler/solvers OK")

    # 2) end-to-end with tiny random models
    vae = CausalVideoVAE(VAEConfig(latent_channels=4, base_channels=16, channel_mult=(1, 2, 2, 4), num_res_blocks=1))
    dit = VideoDiT(DiTConfig(in_channels=4, hidden_size=64, depth=2, num_heads=4, text_dim=32))
    nn.init.normal_(dit.final.linear.weight, std=0.02)
    for blk in dit.blocks:
        nn.init.normal_(blk.cross.proj.weight, std=0.02)
    tcfg = TextEncoderConfig(kind="dummy", embed_dim=32, max_length=8, dtype="float32")
    pipe = VideoPipeline(vae, dit, build_text_encoder(tcfg), device="cpu", dtype=torch.float32)
    kw = dict(num_frames=9, height=64, width=64, num_steps=3, guidance_scale=3.0, guidance_rescale=0.5,
              solver="ab2", seed=1, progress=False)
    a = pipe("a cat on a skateboard", **kw)
    assert a.frames.shape == (1, 9, 64, 64, 3) and a.frames.dtype == torch.uint8
    b = pipe("a cat on a skateboard", **kw)
    assert torch.equal(a.frames, b.frames), "same seed must reproduce"
    c = pipe(["a cat on a skateboard", "a dog"], **kw)
    assert torch.equal(a.frames[0], c.frames[0]), "sample must not depend on batch composition"
    print("pipeline end-to-end OK")

    # 3) save / load round trip
    with tempfile.TemporaryDirectory() as d:
        pipe.save_pretrained(d, tcfg)
        pipe2 = VideoPipeline.from_pretrained(d, device="cpu", dtype=torch.float32)
        e = pipe2("a cat on a skateboard", **kw)
        assert torch.equal(a.frames, e.frames), "reloaded pipeline differs"
    print("save/load round trip OK")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--ckpt", type=str)
    p.add_argument("--prompt", action="append")
    p.add_argument("--negative", type=str, default="")
    p.add_argument("--out", type=str, default="out.mp4")
    p.add_argument("--frames", type=int, default=121)
    p.add_argument("--height", type=int, default=704)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--cfg", type=float, default=5.0)
    p.add_argument("--rescale", type=float, default=0.0)
    p.add_argument("--solver", default="euler", choices=["euler", "ab2", "heun"])
    p.add_argument("--shift", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--offload", action="store_true")
    p.add_argument("--compile", action="store_true")
    a = p.parse_args()
    if a.selftest:
        return _selftest()
    if not (a.ckpt and a.prompt):
        p.error("--ckpt and --prompt are required (or use --selftest)")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[a.dtype]
    pipe = VideoPipeline.from_pretrained(a.ckpt, dtype=dtype, offload=a.offload)
    if a.compile:
        pipe.compile_blocks()
    out = pipe(a.prompt, a.negative, a.frames, a.height, a.width, a.steps, a.cfg, a.rescale,
               a.solver, a.shift, a.seed)
    stem, ext = os.path.splitext(a.out)
    for i in range(len(a.prompt)):
        path = a.out if len(a.prompt) == 1 else f"{stem}_{i}{ext}"
        out.save(path, a.fps, i)
        print(f"saved {path} (seed {out.seed + i})")


if __name__ == "__main__":
    main()
