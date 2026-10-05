"""
train.py — scalable training for the video DiT (rectified flow / flow matching).

Backends
  single      one process (CPU or one GPU)
  ddp         torchrun + DistributedDataParallel; fp32 master weights + bf16/fp16 autocast
  deepspeed   DeepSpeed ZeRO (default stage 3), bf16/fp16, optional CPU/NVMe offload

Data modes
  latents     (recommended at scale) precomputed VAE moments + text embeddings, see `--mode precompute`
  raw         decode video -> VAE -> DiT online (VAE and text encoder run frozen on every GPU)
  synthetic   fake data for smoke tests

Objective (matches pipeline.py exactly)
  x_t = (1 - t) * x0 + t * noise,   target v = noise - x0,   loss = MSE(DiT(x_t, t, text), v)
  t = shift_time(sigmoid(N(t_mean, t_std)), shift)    shift = auto_shift(num_tokens) unless --train_shift > 0
  text is replaced by the null (empty-prompt) embedding with prob. cond_dropout, which enables CFG at inference.

Workflow
  1) python train.py --mode precompute --manifest data.jsonl --data_dir cache --vae_ckpt ckpt/vae \
         --text_kind t5 --text_name google/t5-v1_1-xxl --num_frames 121 --height 704 --width 1280 --batch_size 2
     (launch with torchrun --nproc_per_node=8 to shard; resumable; manifest = jsonl {"video": path, "caption": str})
  2) python train.py --mode stats --data_dir cache            # writes cache/latent_stats.json (shift/scale)
  3) torchrun --nproc_per_node=8 train.py --backend ddp --data_mode latents --data_dir cache \
         --vae_ckpt ckpt/vae --dit_preset large --output_dir runs/exp1 --max_steps 200000
     deepspeed --num_gpus=8 train.py --backend deepspeed --zero_stage 3 ...   (torchrun works too)
  4) runs/exp1/export/ is a directory pipeline.py loads directly (EMA weights when EMA is enabled).

Any option can also come from --config file.json (CLI flags override). Smoke test: python train.py --selftest
Not implemented: aspect-ratio bucketing (latents in one run must share a shape), sequence parallelism.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import glob
import json
import math
import os
import random
import re
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Sampler, Subset

from dit_blocks import DiTConfig, VideoDiT
from pipeline import auto_shift, config_from_dict, load_component, save_component, shift_time, write_pipeline_index
from text_encoder import BaseTextEncoder, TextEncoderConfig, build_text_encoder
from vae_model import CausalVideoVAE, VAEConfig

try:
    import deepspeed

    HAS_DEEPSPEED = True
except Exception:  # pragma: no cover
    deepspeed = None
    HAS_DEEPSPEED = False


# =========================================================================== #
# Config
# =========================================================================== #
DIT_PRESETS: Dict[str, Dict[str, Any]] = {
    "tiny": dict(hidden_size=64, depth=2, num_heads=4),        # tests
    "small": dict(hidden_size=768, depth=12, num_heads=12),    # ~0.2B
    "base": dict(hidden_size=1152, depth=28, num_heads=16),    # ~0.6B
    "large": dict(hidden_size=2048, depth=32, num_heads=16),   # ~2B
    "xl": dict(hidden_size=3072, depth=40, num_heads=24),      # ~6B
}


@dataclass
class TrainConfig:
    # ---- io ----
    output_dir: str = "runs/exp"
    data_mode: str = "latents"          # latents | raw | synthetic
    data_dir: str = ""                  # latents mode / precompute output (samples/, null.pt, latent_stats.json)
    manifest: str = ""                  # raw mode / precompute: jsonl {"video","caption"}
    video_root: str = ""
    vae_ckpt: str = ""                  # component dir with config.json + weights (as written by pipeline.save_component)
    # ---- clip geometry (raw / precompute / synthetic) ----
    num_frames: int = 121               # must be 1 + 4k
    height: int = 704
    width: int = 1280
    frame_stride: int = 1
    synthetic_size: int = 256
    # ---- model ----
    dit_preset: str = "base"
    dit_overrides: dict = dataclasses.field(default_factory=dict)
    grad_checkpointing: bool = True
    compile: bool = False
    # ---- text encoder (raw / precompute / export) ----
    text_kind: str = "t5"
    text_name: str = "google/t5-v1_1-xxl"
    text_max_length: int = 256
    text_dtype: str = "bfloat16"
    text_embed_dim: int = 64            # dummy only
    text_extra: dict = dataclasses.field(default_factory=dict)   # extra TextEncoderConfig fields (llm settings)
    # ---- objective ----
    cond_dropout: float = 0.1
    t_sampling: str = "logit_normal"    # logit_normal | uniform
    t_mean: float = 0.0
    t_std: float = 1.0
    train_shift: float = 0.0            # <= 0: auto from token count (same rule as inference)
    # ---- optimisation ----
    backend: str = "ddp"                # single | ddp | deepspeed
    precision: str = "bf16"             # bf16 | fp16 | fp32
    batch_size: int = 1                 # per GPU micro-batch
    grad_accum: int = 1
    max_steps: int = 100000
    lr: float = 1e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0
    ema_decay: float = 0.9999           # 0 disables EMA
    ema_every: int = 1
    # ---- deepspeed ----
    zero_stage: int = 3
    offload_optimizer: str = "none"     # none | cpu | nvme
    offload_param: str = "none"         # none | cpu | nvme
    nvme_path: str = "/local_nvme"
    ds_persistence_threshold: int = 100000
    ds_config: str = ""                 # path to a full DeepSpeed json (overrides the generated one)
    # ---- loop ----
    seed: int = 0
    num_workers: int = 4
    log_every: int = 10
    save_every: int = 1000
    keep_last: int = 3
    export_every: int = 0               # 0 = export only at the end
    export_dtype: str = "bfloat16"
    resume: str = "auto"                # auto | none | path to a checkpoints/step_XXXXXXXX dir
    peak_tflops: float = 0.0            # per-GPU peak for the approximate MFU readout (0 = off)
    wandb: bool = False
    wandb_project: str = "video-dit"
    wandb_name: str = ""

    def validate(self) -> None:
        assert self.backend in ("single", "ddp", "deepspeed"), self.backend
        assert self.data_mode in ("latents", "raw", "synthetic"), self.data_mode
        assert self.precision in ("bf16", "fp16", "fp32"), self.precision
        assert self.t_sampling in ("logit_normal", "uniform"), self.t_sampling
        assert (self.num_frames - 1) % 4 == 0, "num_frames must be 1 + 4k"
        assert self.grad_accum >= 1 and self.batch_size >= 1
        assert 0.0 <= self.cond_dropout < 1.0
        if self.backend == "deepspeed" and not HAS_DEEPSPEED:
            raise RuntimeError("backend=deepspeed requested but `deepspeed` is not installed")


def text_config_from(cfg: TrainConfig) -> TextEncoderConfig:
    d = dict(kind=cfg.text_kind, name=cfg.text_name, max_length=cfg.text_max_length,
             dtype=cfg.text_dtype, embed_dim=cfg.text_embed_dim)
    d.update(cfg.text_extra)
    return config_from_dict(TextEncoderConfig, d)


def read_vae_config(cfg: TrainConfig) -> VAEConfig:
    if cfg.vae_ckpt:
        with open(os.path.join(cfg.vae_ckpt, "config.json")) as f:
            return config_from_dict(VAEConfig, json.load(f))
    return VAEConfig()


def build_dit_config(cfg: TrainConfig, vae_cfg: VAEConfig) -> DiTConfig:
    d = dict(DIT_PRESETS.get(cfg.dit_preset, {}))
    d["in_channels"] = vae_cfg.latent_channels
    d["text_dim"] = cfg.text_embed_dim if cfg.text_kind == "dummy" else 4096
    d.update(cfg.dit_overrides)
    d["gradient_checkpointing"] = cfg.grad_checkpointing
    return config_from_dict(DiTConfig, d)


def latent_dims(cfg: TrainConfig, vae_cfg: VAEConfig) -> Tuple[int, int, int]:
    tr = 2 ** sum(vae_cfg.temporal_down)
    sr = 2 ** (len(vae_cfg.channel_mult) - 1)
    return (cfg.num_frames - 1) // tr + 1, cfg.height // sr, cfg.width // sr


# =========================================================================== #
# Distributed helpers
# =========================================================================== #
@dataclass
class DistInfo:
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    device: torch.device = torch.device("cpu")

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def init_distributed(timeout_min: int = 60) -> DistInfo:
    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local = int(os.environ.get("LOCAL_RANK", 0))
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.set_device(local)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if world > 1 and not dist.is_initialized():
        dist.init_process_group("nccl" if cuda else "gloo", timeout=timedelta(minutes=timeout_min))
    return DistInfo(rank, local, world, torch.device("cuda", local) if cuda else torch.device("cpu"))


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def all_reduce_mean(x: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        x = x.clone()
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        x /= dist.get_world_size()
    return x


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextlib.contextmanager
def default_dtype(dtype: torch.dtype):
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)


# =========================================================================== #
# Logging
# =========================================================================== #
class Logger:
    def __init__(self, cfg: TrainConfig, is_main: bool):
        self.is_main = is_main
        self.file = None
        self.wandb = None
        if not is_main:
            return
        os.makedirs(cfg.output_dir, exist_ok=True)
        self.file = open(os.path.join(cfg.output_dir, "metrics.jsonl"), "a", buffering=1)
        if cfg.wandb:
            try:
                import wandb

                wandb.init(project=cfg.wandb_project, name=cfg.wandb_name or None, config=dataclasses.asdict(cfg))
                self.wandb = wandb
            except Exception as e:  # pragma: no cover
                print(f"[warn] wandb disabled: {e}")

    def info(self, msg: str) -> None:
        if self.is_main:
            print(msg, flush=True)

    def log(self, step: int, metrics: Dict[str, float]) -> None:
        if not self.is_main:
            return
        rec = {"step": step, **metrics}
        self.file.write(json.dumps(rec) + "\n")
        if self.wandb is not None:
            self.wandb.log(metrics, step=step)
        body = " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items())
        print(f"[step {step}] {body}", flush=True)


# =========================================================================== #
# Data
# =========================================================================== #
def resize_crop(frames: torch.Tensor, h: int, w: int, random_crop: bool) -> torch.Tensor:
    """uint8 [T,H0,W0,3] -> uint8 [T,h,w,3]; cover-resize then crop."""
    T, H0, W0, _ = frames.shape
    x = frames.permute(0, 3, 1, 2).float()
    scale = max(h / H0, w / W0)
    nh, nw = max(h, round(H0 * scale)), max(w, round(W0 * scale))
    if (nh, nw) != (H0, W0):
        x = F.interpolate(x, size=(nh, nw), mode="bilinear", antialias=True, align_corners=False)
    top = random.randint(0, nh - h) if random_crop else (nh - h) // 2
    left = random.randint(0, nw - w) if random_crop else (nw - w) // 2
    x = x[:, :, top: top + h, left: left + w]
    return x.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous()


def read_clip(path: str, num_frames: int, stride: int) -> torch.Tensor:
    """Random contiguous clip as uint8 [T,H,W,3]."""
    need = (num_frames - 1) * stride + 1
    try:
        import decord
    except ImportError:
        decord = None
    if decord is not None:
        vr = decord.VideoReader(path, num_threads=1)
        n = len(vr)
        if n < need:
            raise ValueError(f"{path}: {n} frames < required {need}")
        start = random.randint(0, n - need)
        idx = list(range(start, start + need, stride))
        return torch.from_numpy(vr.get_batch(idx).asnumpy())
    from torchvision.io import read_video  # fallback: decodes the whole file

    vid, _, _ = read_video(path, pts_unit="sec", output_format="THWC")
    if vid.shape[0] < need:
        raise ValueError(f"{path}: {vid.shape[0]} frames < required {need}")
    start = random.randint(0, vid.shape[0] - need)
    return vid[start: start + need: stride].contiguous()


class RawVideoDataset(Dataset):
    def __init__(self, cfg: TrainConfig, random_crop: bool):
        with open(cfg.manifest) as f:
            self.items = [json.loads(l) for l in f if l.strip()]
        if not self.items:
            raise RuntimeError(f"empty manifest: {cfg.manifest}")
        self.cfg, self.random_crop = cfg, random_crop

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        c, last_err = self.cfg, None
        for attempt in range(10):
            j = i if attempt == 0 else random.randrange(len(self.items))
            try:
                it = self.items[j]
                frames = read_clip(os.path.join(c.video_root, it["video"]), c.num_frames, c.frame_stride)
                frames = resize_crop(frames, c.height, c.width, self.random_crop)
                return {"video": frames, "caption": it["caption"], "index": j}
            except Exception as e:  # corrupt / short video: fall back to another sample
                last_err = e
        raise RuntimeError(f"10 consecutive failures reading samples, last error: {last_err}")


def _collate_text(items: List[dict]) -> Tuple[torch.Tensor, torch.Tensor]:
    L = max(it["emb"].shape[0] for it in items)
    D = items[0]["emb"].shape[1]
    emb = torch.zeros(len(items), L, D, dtype=items[0]["emb"].dtype)
    mask = torch.zeros(len(items), L, dtype=torch.bool)
    for i, it in enumerate(items):
        n = it["emb"].shape[0]
        emb[i, :n] = it["emb"]
        mask[i, :n] = True
    return emb, mask


def collate_raw(items: List[dict]) -> dict:
    return {"video": torch.stack([it["video"] for it in items]),
            "caption": [it["caption"] for it in items],
            "index": [it["index"] for it in items]}


def collate_latent(items: List[dict]) -> dict:
    shapes = {tuple(it["moments"].shape) for it in items}
    if len(shapes) != 1:
        raise RuntimeError(f"latents in a batch must share a shape (no bucketing implemented), got {shapes}")
    emb, mask = _collate_text(items)
    return {"moments": torch.stack([it["moments"] for it in items]), "emb": emb, "mask": mask}


class LatentDataset(Dataset):
    def __init__(self, data_dir: str):
        self.files = sorted(glob.glob(os.path.join(data_dir, "samples", "*.pt")))
        if not self.files:
            raise RuntimeError(f"no precomputed samples in {data_dir}/samples (run --mode precompute first)")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int) -> dict:
        d = torch.load(self.files[i], map_location="cpu", weights_only=True)
        return {"moments": d["moments"], "emb": d["emb"]}


class SyntheticDataset(Dataset):
    """A fixed latent pattern plus tiny noise; captions from a small vocabulary (learnable => loss must drop)."""

    VOCAB = ["a", "red", "blue", "fox", "dog", "runs", "jumps", "over", "the", "hill"]

    def __init__(self, cfg: TrainConfig, vae_cfg: VAEConfig, text_dim: int):
        self.n = cfg.synthetic_size
        T, H, W = latent_dims(cfg, vae_cfg)
        g = torch.Generator().manual_seed(1234)
        self.pattern = torch.randn(vae_cfg.latent_channels, T, H, W, generator=g)
        self.enc = build_text_encoder(TextEncoderConfig(kind="dummy", embed_dim=text_dim, max_length=8, dtype="float32"))

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> dict:
        g = torch.Generator().manual_seed(i)
        mean = self.pattern + 0.01 * torch.randn(self.pattern.shape, generator=g)
        moments = torch.cat([mean, torch.full_like(mean, -12.0)], dim=0)
        words = [self.VOCAB[int(k)] for k in torch.randint(0, len(self.VOCAB), (3,), generator=g)]
        emb, mask = self.enc(" ".join(words))
        return {"moments": moments, "emb": emb[0][mask[0]]}


class ResumableSampler(Sampler):
    """Rank-sharded shuffled sampler that can restart mid-epoch (start = per-rank samples already consumed)."""

    def __init__(self, n: int, rank: int, world: int, seed: int, shuffle: bool = True):
        self.n, self.rank, self.world, self.seed, self.shuffle = n, rank, world, seed, shuffle
        self.epoch, self.start = 0, 0

    def set_state(self, epoch: int, start: int) -> None:
        self.epoch, self.start = epoch, start

    def _indices(self) -> List[int]:
        if self.shuffle:
            g = torch.Generator().manual_seed(self.seed + self.epoch)
            idx = torch.randperm(self.n, generator=g).tolist()
        else:
            idx = list(range(self.n))
        total = math.ceil(self.n / self.world) * self.world
        idx += idx[: total - self.n]
        return idx[self.rank:total:self.world]

    def __iter__(self):
        return iter(self._indices()[self.start:])

    def __len__(self) -> int:
        return max(0, math.ceil(self.n / self.world) - self.start)


# =========================================================================== #
# EMA
# =========================================================================== #
class EMA:
    """fp32 shadow of a list of tensors (full params for single/ddp, local ZeRO-3 shards for deepspeed)."""

    def __init__(self, names: List[str], tensors: Sequence[torch.Tensor], decay: float):
        self.names, self.decay = list(names), decay
        self.shadow = [t.detach().float().clone() for t in tensors]

    @torch.no_grad()
    def update(self, tensors: Sequence[torch.Tensor], step: int) -> None:
        d = min(self.decay, (1.0 + step) / (10.0 + step))
        src = [t.detach() if t.dtype == torch.float32 else t.detach().float() for t in tensors]
        torch._foreach_lerp_(self.shadow, src, 1.0 - d)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {n: s.detach().cpu() for n, s in zip(self.names, self.shadow)}

    def load_state_dict(self, sd: Dict[str, torch.Tensor]) -> None:
        for n, s in zip(self.names, self.shadow):
            s.copy_(sd[n].to(s.device, s.dtype))


def _ds_view(p: nn.Parameter) -> torch.Tensor:
    """Tensor holding this rank's live copy of a ZeRO-3 parameter: the full data for persistent params,
    otherwise the local partition."""
    return p.data if getattr(p, "ds_persist", False) else p.ds_tensor


# =========================================================================== #
# Trainer
# =========================================================================== #
@dataclass
class TrainState:
    step: int = 0
    epoch: int = 0
    consumed: int = 0      # per-rank samples consumed in the current epoch
    skipped: int = 0


class Trainer:
    def __init__(self, cfg: TrainConfig):
        cfg.validate()
        self.cfg = cfg
        self.dist = init_d