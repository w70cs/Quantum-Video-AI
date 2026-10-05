"""FastAPI backend for the AI video generation service.

Run (single process only: one process owns the GPU and the in-memory queue):

    uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1

Environment variables (all optional):
    PIPELINE_TARGET      "module:ClassName" of the pipeline in /ai_core
                         (default "ai_core.pipeline:VideoGenerationPipeline")
    CHECKPOINT_DIR       model weights directory               (default ./checkpoints)
    DEVICE / DTYPE       cuda|cuda:0|cpu / bfloat16|float16|float32
    OUTPUT_DIR           where finished mp4 files are stored   (default ./outputs)
    CORS_ORIGINS         comma-separated origins or "*"        (default "*")
    API_KEYS             comma-separated keys; empty = auth disabled
    MAX_QUEUE_SIZE       max waiting jobs                      (default 32)
    MAX_JOBS_PER_CLIENT  max queued+running jobs per client    (default 3)
    JOB_TIMEOUT_SECONDS  per-job wall-clock limit              (default 900)
    JOB_TTL_SECONDS      keep finished jobs/videos this long   (default 3600)
    MAX_FRAMES           hard cap on frames per video          (default 241)
    TEMPORAL_STRIDE      causal VAE temporal compression       (default 4)
    SPATIAL_MULTIPLE     VAE spatial downsample x patch size   (default 16)
"""
from __future__ import annotations

import asyncio
import gc
import importlib
import inspect
import logging
import os
import secrets
import threading
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import APIKeyHeader

from schemas import (
    ErrorResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    JobStatus,
    JobStatusResponse,
    QueueStatsResponse,
    ResolvedParams,
    VideoInfo,
)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(threadName)s] %(name)s: %(message)s",
)
logger = logging.getLogger("videogen.api")


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
def _env_list(name: str, default: str = "") -> list[str]:
    return [x.strip() for x in os.getenv(name, default).split(",") if x.strip()]


@dataclass(frozen=True)
class Settings:
    pipeline_target: str
    checkpoint_dir: str
    device: str
    dtype: str
    output_dir: Path
    cors_origins: list[str]
    api_keys: list[str]
    max_queue_size: int
    max_jobs_per_client: int
    job_timeout_seconds: int
    job_ttl_seconds: int
    max_frames: int
    temporal_stride: int
    spatial_multiple: int

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            pipeline_target=os.getenv("PIPELINE_TARGET", "ai_core.pipeline:VideoGenerationPipeline"),
            checkpoint_dir=os.getenv("CHECKPOINT_DIR", "./checkpoints"),
            device=os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"),
            dtype=os.getenv("DTYPE", "bfloat16"),
            output_dir=Path(os.getenv("OUTPUT_DIR", "./outputs")),
            cors_origins=_env_list("CORS_ORIGINS", "*"),
            api_keys=_env_list("API_KEYS"),
            max_queue_size=int(os.getenv("MAX_QUEUE_SIZE", "32")),
            max_jobs_per_client=int(os.getenv("MAX_JOBS_PER_CLIENT", "3")),
            job_timeout_seconds=int(os.getenv("JOB_TIMEOUT_SECONDS", "900")),
            job_ttl_seconds=int(os.getenv("JOB_TTL_SECONDS", "3600")),
            max_frames=int(os.getenv("MAX_FRAMES", "241")),
            temporal_stride=int(os.getenv("TEMPORAL_STRIDE", "4")),
            spatial_multiple=int(os.getenv("SPATIAL_MULTIPLE", "16")),
        )


settings = Settings.from_env()


# --------------------------------------------------------------------------- #
# Job model + thread-safe store
# --------------------------------------------------------------------------- #
class JobAborted(Exception):
    """Raised inside the inference thread to stop a cancelled / timed-out job."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason  # "cancelled" | "timeout"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Job:
    id: str
    request: GenerateRequest
    params: ResolvedParams
    client_id: str
    seed: int
    status: JobStatus = JobStatus.QUEUED
    progress: float = 0.0
    created_at: datetime = field(default_factory=_utcnow)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    video: Optional[VideoInfo] = None
    video_path: Optional[Path] = None
    error: Optional[str] = None
    deadline: float = float("inf")
    cancel_event: threading.Event = field(default_factory=threading.Event)


class JobStore:
    """All mutations go through an RLock because the GPU thread updates progress
    while the event loop serves status requests."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._pending: deque[str] = deque()
        self._lock = threading.RLock()

    def add(self, job: Job) -> None:
        with self._lock:
            self._jobs[job.id] = job
            self._pending.append(job.id)

    def discard(self, job_id: str) -> None:
        with self._lock:
            self._jobs.pop(job_id, None)
            try:
                self._pending.remove(job_id)
            except ValueError:
                pass

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def position(self, job_id: str) -> int:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return 0
            if job.status is JobStatus.RUNNING:
                return 0
            if job.status is JobStatus.QUEUED:
                try:
                    return list(self._pending).index(job_id) + 1
                except ValueError:
                    return 0
            return 0

    def start(self, job_id: str, timeout_s: int) -> Optional[Job]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status is not JobStatus.QUEUED:
                return None
            try:
                self._pending.remove(job_id)
            except ValueError:
                pass
            job.status = JobStatus.RUNNING
            job.started_at = _utcnow()
            job.deadline = time.monotonic() + timeout_s
            return job

    def set_progress(self, job_id: str, value: float) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job and job.status is JobStatus.RUNNING:
                job.progress = max(job.progress, min(max(value, 0.0), 0.99))

    def finish(
        self,
        job_id: str,
        status: JobStatus,
        *,
        error: Optional[str] = None,
        video: Optional[VideoInfo] = None,
        video_path: Optional[Path] = None,
    ) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = status
            job.error = error
            job.video = video
            job.video_path = video_path
            job.finished_at = _utcnow()
            if status is JobStatus.COMPLETED:
                job.progress = 1.0

    def cancel_queued(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status is not JobStatus.QUEUED:
                return False
            try:
                self._pending.remove(job_id)
            except ValueError:
                pass
            job.status = JobStatus.CANCELLED
            job.finished_at = _utcnow()
            job.error = "Cancelled by user."
            job.cancel_event.set()
            return True

    def active_for(self, client_id: str) -> int:
        with self._lock:
            return sum(
                1
                for j in self._jobs.values()
                if j.client_id == client_id and not j.status.is_terminal
            )

    def counts(self) -> tuple[int, int, int]:
        with self._lock:
            running = sum(1 for j in self._jobs.values() if j.status is JobStatus.RUNNING)
            return len(self._pending), running, len(self._jobs)

    def purge(self, ttl_s: int) -> list[Path]:
        cutoff = _utcnow().timestamp() - ttl_s
        paths: list[Path] = []
        with self._lock:
            for jid in [
                jid
                for jid, j in self._jobs.items()
                if j.status.is_terminal and j.finished_at and j.finished_at.timestamp() < cutoff
            ]:
                job = self._jobs.pop(jid)
                if job.video_path:
                    paths.append(job.video_path)
        return paths


# --------------------------------------------------------------------------- #
# Pipeline loading + inference helpers
# --------------------------------------------------------------------------- #
def _param_names(fn: Callable[..., Any]) -> Optional[set[str]]:
    """Explicit parameter names of `fn`, or None if it takes **kwargs / is opaque."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return None
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return None
    return set(params)


def _call_filtered(fn: Callable[..., Any], **kwargs: Any) -> Any:
    names = _param_names(fn)
    if names is None:
        return fn(**kwargs)
    return fn(**{k: v for k, v in kwargs.items() if k in names})


def _pick_name(fn: Callable[..., Any], candidates: tuple[str, ...]) -> Optional[str]:
    try:
        names = set(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        return None
    return next((c for c in candidates if c in names), None)


def load_pipeline(cfg: Settings) -> Any:
    module_name, _, attr = cfg.pipeline_target.partition(":")
    if not module_name or not attr:
        raise RuntimeError("PIPELINE_TARGET must look like 'package.module:ClassName'")
    cls = getattr(importlib.import_module(module_name), attr)
    dtype = getattr(torch, cfg.dtype)
    logger.info("Loading pipeline %s from %s on %s (%s)", cfg.pipeline_target, cfg.checkpoint_dir, cfg.device, cfg.dtype)
    t0 = time.perf_counter()
    factory = cls.from_pretrained if hasattr(cls, "from_pretrained") else cls
    pipe = _call_filtered(
        factory,
        checkpoint_dir=cfg.checkpoint_dir,
        pretrained_model_name_or_path=cfg.checkpoint_dir,
        device=cfg.device,
        dtype=dtype,
        torch_dtype=dtype,
    )
    if hasattr(pipe, "to") and not hasattr(cls, "from_pretrained"):
        pipe = pipe.to(cfg.device)
    if hasattr(pipe, "eval"):
        pipe.eval()
    logger.info("Pipeline ready in %.1fs", time.perf_counter() - t0)
    return pipe


def _extract_video(out: Any) -> torch.Tensor:
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, dict):
        for key in ("video", "videos", "frames", "sample", "samples"):
            if key in out:
                return _extract_video(out[key])
    for attr in ("video", "videos", "frames", "sample", "samples"):
        if hasattr(out, attr):
            return _extract_video(getattr(out, attr))
    if isinstance(out, (list, tuple)) and out:
        return _extract_video(out[0])
    raise TypeError(f"Unsupported pipeline output type: {type(out)!r}")


def _to_frames(video: torch.Tensor) -> np.ndarray:
    """Normalize (B,C,T,H,W) | (C,T,H,W) | (T,H,W,C) in [-1,1]/[0,1]/uint8 -> uint8 (T,H,W,3)."""
    v = video.detach().cpu()
    if v.ndim == 5:
        v = v[0]
    if v.ndim != 4:
        raise ValueError(f"Expected a 4D/5D video tensor, got shape {tuple(v.shape)}")
    if v.shape[0] == 3 and v.shape[-1] != 3:
        v = v.permute(1, 2, 3, 0)
    elif v.shape[-1] != 3:
        raise ValueError(f"Cannot locate RGB channel axis in shape {tuple(v.shape)}")
    if v.dtype != torch.uint8:
        v = v.float()
        if float(v.min()) < 0.0:
            v = (v + 1.0) / 2.0
        v = (v.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
    return v.contiguous().numpy()


def _encode_mp4(frames: np.ndarray, fps: int, dest: Path) -> None:
    import imageio.v2 as imageio  # requires: imageio, imageio-ffmpeg

    tmp = dest.with_suffix(".part.mp4")
    try:
        with imageio.get_writer(
            str(tmp),
            format="FFMPEG",
            fps=fps,
            codec="libx264",
            quality=8,
            pixelformat="yuv420p",
            macro_block_size=1,
            ffmpeg_params=["-movflags", "+faststart"],
        ) as writer:
            for frame in frames:
                writer.append_data(frame)
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)


def _check_abort(job: Job) -> None:
    if job.cancel_event.is_set():
        raise JobAborted("cancelled")
    if time.monotonic() > job.deadline:
        raise JobAborted("timeout")


def _generate_video(core: "Core", job: Job) -> VideoInfo:
    """Runs on the dedicated GPU thread. Blocking."""
    cfg, pipe, store = core.cfg, core.pipeline, core.store
    t0 = time.perf_counter()
    _check_abort(job)

    def on_step(step: int, total: int) -> None:
        _check_abort(job)
        store.set_progress(job.id, 0.02 + 0.88 * (step / max(total, 1)))

    fn = getattr(pipe, "generate", None) or pipe
    kwargs: dict[str, Any] = dict(
        prompt=job.request.prompt,
        negative_prompt=job.request.negative_prompt,
        height=job.params.height,
        width=job.params.width,
        num_frames=job.params.num_frames,
        fps=job.params.fps,
        num_inference_steps=job.request.num_inference_steps,
        guidance_scale=job.request.guidance_scale,
    )
    seed_name = _pick_name(fn, ("generator", "seed"))
    if seed_name == "generator":
        gen_device = "cpu" if cfg.device == "cpu" else cfg.device
        kwargs["generator"] = torch.Generator(device=gen_device).manual_seed(job.seed)
    elif seed_name == "seed":
        kwargs["seed"] = job.seed
    cb_name = _pick_name(fn, ("callback", "progress_callback"))
    if cb_name:
        kwargs[cb_name] = on_step

    with torch.inference_mode():
        out = _call_filtered(fn, **kwargs)

    _check_abort(job)
    store.set_progress(job.id, 0.92)
    frames = _to_frames(_extract_video(out))
    del out

    _check_abort(job)
    dest = cfg.output_dir / f"{job.id}.mp4"
    _encode_mp4(frames, job.params.fps, dest)
    job.video_path = dest

    return VideoInfo(
        url=f"/api/videos/{job.id}",
        width=int(frames.shape[2]),
        height=int(frames.shape[1]),
        num_frames=int(frames.shape[0]),
        fps=job.params.fps,
        duration_seconds=round(frames.shape[0] / job.params.fps, 3),
        size_bytes=dest.stat().st_size,
        seed=job.seed,
        generation_seconds=round(time.perf_counter() - t0, 2),
    )


# --------------------------------------------------------------------------- #
# Core runtime: queue + worker + janitor
# --------------------------------------------------------------------------- #
class Core:
    def __init__(self, cfg: Settings) -> None:
        self.cfg = cfg
        self.store = JobStore()
        self.queue: asyncio.Queue[str] = asyncio.Queue(maxsize=cfg.max_queue_size)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu")
        self.pipeline: Any = None
        self.ready = False
        self.load_error: Optional[str] = None
        self.current_job_id: Optional[str] = None
        self.started_monotonic = time.monotonic()
        self.tasks: list[asyncio.Task] = []

    async def worker(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            job_id = await self.queue.get()
            try:
                job = self.store.start(job_id, self.cfg.job_timeout_seconds)
                if job is None:  # cancelled while waiting
                    continue
                self.current_job_id = job.id
                await self._run(loop, job)
            except asyncio.CancelledError:
                raise
            except Exception:  # never let the worker die
                logger.exception("Unexpected worker error on job %s", job_id)
            finally:
                self.current_job_id = None
                self.queue.task_done()

    async def _run(self, loop: asyncio.AbstractEventLoop, job: Job) -> None:
        logger.info("Job %s started (%dx%d, %d frames)", job.id, job.params.width, job.params.height, job.params.num_frames)
        try:
            info = await loop.run_in_executor(self.executor, _generate_video, self, job)
            self.store.finish(job.id, JobStatus.COMPLETED, video=info, video_path=job.video_path)
            logger.info("Job %s completed in %.1fs", job.id, info.generation_seconds)
        except JobAborted as exc:
            if exc.reason == "timeout":
                self.store.finish(job.id, JobStatus.FAILED, error="Generation timed out.")
            else:
                self.store.finish(job.id, JobStatus.CANCELLED, error="Cancelled by user.")
            logger.info("Job %s aborted: %s", job.id, exc.reason)
        except torch.cuda.OutOfMemoryError:
            logger.error("Job %s hit CUDA OOM", job.id)
            self.store.finish(
                job.id,
                JobStatus.FAILED,
                error="GPU out of memory. Try a lower resolution, shorter duration, or fewer frames.",
            )
        except Exception:
            logger.exception("Job %s failed", job.id)
            self.store.finish(job.id, JobStatus.FAILED, error="Generation failed due to an internal error.")
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    async def janitor(self) -> None:
        while True:
            await asyncio.sleep(60)
            try:
                for path in self.store.purge(self.cfg.job_ttl_seconds):
                    await asyncio.to_thread(path.unlink, True)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Janitor error")


# --------------------------------------------------------------------------- #
# App lifecycle
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    core = Core(settings)
    app.state.core = core
    settings.output_dir.mkdir(parents=True, exist_ok=True)

    async def _boot() -> None:
        try:
            core.pipeline = await asyncio.to_thread(load_pipeline, settings)
            core.ready = True
        except Exception as exc:
            logger.exception("Pipeline failed to load")
            core.load_error = str(exc)

    core.tasks = [
        asyncio.create_task(_boot(), name="boot"),
        asyncio.create_task(core.worker(), name="worker"),
        asyncio.create_task(core.janitor(), name="janitor"),
    ]
    try:
        yield
    finally:
        running = core.store.get(core.current_job_id) if core.current_job_id else None
        if running:
            running.cancel_event.set()
        for t in core.tasks:
            t.cancel()
        await asyncio.gather(*core.tasks, return_exceptions=True)
        core.executor.shutdown(wait=False, cancel_futures=True)
