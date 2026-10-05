"""Pydantic v2 schemas for the AI video generation API."""
from __future__ import annotations

import re
from datetime import datetime
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class AspectRatio(str, Enum):
    LANDSCAPE_16_9 = "16:9"
    PORTRAIT_9_16 = "9:16"
    SQUARE_1_1 = "1:1"
    LANDSCAPE_4_3 = "4:3"
    PORTRAIT_3_4 = "3:4"
    CINEMATIC_21_9 = "21:9"

    @property
    def ratio(self) -> tuple[int, int]:
        w, h = self.value.split(":")
        return int(w), int(h)


class Resolution(str, Enum):
    """Short-side resolution preset."""

    P480 = "480p"
    P720 = "720p"

    @property
    def short_side(self) -> int:
        return int(self.value[:-1])


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED)


# --------------------------------------------------------------------------- #
# Request
# --------------------------------------------------------------------------- #
class ResolvedParams(BaseModel):
    """Concrete tensor-level parameters derived from a GenerateRequest."""

    width: int
    height: int
    num_frames: int
    fps: int
    duration_seconds: float


class GenerateRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "prompt": "A red fox running through a snowy forest at golden hour, cinematic",
                "negative_prompt": "blurry, low quality, watermark",
                "aspect_ratio": "16:9",
                "resolution": "480p",
                "fps": 24,
                "duration_seconds": 4.0,
                "num_inference_steps": 50,
                "guidance_scale": 7.5,
                "seed": 42,
            }
        },
    )

    prompt: str = Field(..., min_length=3, max_length=2000, description="Text description of the video.")
    negative_prompt: Optional[str] = Field(default=None, max_length=1000)
    aspect_ratio: AspectRatio = AspectRatio.LANDSCAPE_16_9
    resolution: Resolution = Resolution.P480
    fps: Literal[8, 12, 16, 24, 30] = 24
    duration_seconds: float = Field(default=4.0, ge=1.0, le=10.0)
    num_inference_steps: int = Field(default=50, ge=10, le=100)
    guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0)
    seed: Optional[int] = Field(default=None, ge=0, le=2**31 - 1)

    @field_validator("prompt", "negative_prompt")
    @classmethod
    def _clean_text(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v = _CONTROL_CHARS.sub("", v).strip()
        return v or None

    @field_validator("prompt")
    @classmethod
    def _prompt_not_empty(cls, v: Optional[str]) -> str:
        if not v or len(v) < 3:
            raise ValueError("prompt must contain at least 3 visible characters")
        return v

    def resolve(self, temporal_stride: int = 4, spatial_multiple: int = 16) -> ResolvedParams:
        """Snap the request to dimensions the VAE/DiT can process.

        - Spatial dims become multiples of `spatial_multiple` (VAE downsample x patch size).
        - Frame count becomes `k * temporal_stride + 1` (causal VAE: first frame is standalone).
        """
        rw, rh = self.aspect_ratio.ratio
        short = self.resolution.short_side
        if rw >= rh:
            h, w = float(short), short * rw / rh
        else:
            w, h = float(short), short * rh / rw

        def snap(v: float) -> int:
            return max(spatial_multiple, int(round(v / spatial_multiple)) * spatial_multiple)

        raw = max(1, round(self.duration_seconds * self.fps))
        k = max(1, round((raw - 1) / temporal_stride))
        num_frames = k * temporal_stride + 1
        return ResolvedParams(
            width=snap(w),
            height=snap(h),
            num_frames=num_frames,
            fps=self.fps,
            duration_seconds=round(num_frames / self.fps, 3),
        )


# --------------------------------------------------------------------------- #
# Responses
# --------------------------------------------------------------------------- #
class GenerateResponse(BaseModel):
    job_id: str
    status: JobStatus
    queue_position: int = Field(..., description="1-based position in queue; 0 if already running.")
    created_at: datetime
    status_url: str
    resolved: ResolvedParams


class VideoInfo(BaseModel):
    url: str
    width: int
    height: int
    num_frames: int
    fps: int
    duration_seconds: float
    size_bytes: int
    seed: int
    generation_seconds: float


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    queue_position: int = 0
    progress: float = Field(0.0, ge=0.0, le=1.0)
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    request: GenerateRequest
    resolved: ResolvedParams
    video: Optional[VideoInfo] = None
    error: Optional[str] = None


class QueueStatsResponse(BaseModel):
    queued: int
    running: int
    max_queue_size: int
    total_tracked_jobs: int


class HealthResponse(BaseModel):
    status: Literal["ok", "loading", "error"]
    model_loaded: bool
    device: str
    queue_size: int
    active_job_id: Optional[str] = None
    uptime_seconds: float


class ErrorResponse(BaseModel):
    detail: str
      
