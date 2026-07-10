"""Pydantic response models for the Aura API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

JobStatus = Literal["queued", "running", "done", "error"]


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    cuda_available: bool
    gpu_name: str | None = None
    vram_used_gb: float | None = None
    vram_total_gb: float | None = None
    editor_loaded: bool
    expander_enabled: bool
    expander_loaded: bool
    loras_loaded: list[str] = Field(default_factory=list)
    metrics_available: bool


class ProcedureInfo(BaseModel):
    name: str
    description: str
    example_instruction: str


class ExpandResponse(BaseModel):
    prompt: str
    procedure: str
    used_fallback: bool
    latency_s: float


class MetricsResponse(BaseModel):
    edit_magnitude: float | None
    arcface_cosine: float | None
    lpips: float | None
    clip_score: float | None
    canary_static: bool = Field(
        description="True = output ≈ input (static-image collapse detected)"
    )


class GenerateResult(BaseModel):
    prompt_used: str
    procedure: str
    seed: int | None
    num_steps: int
    expander_used: bool
    latency_s: float
    metrics: MetricsResponse | None = None


class GenerateResponse(GenerateResult):
    image_b64: str = Field(description="PNG, base64-encoded")


class JobSubmitResponse(BaseModel):
    job_id: str
    status: JobStatus
    status_url: str
    image_url: str


class JobInfo(BaseModel):
    job_id: str
    status: JobStatus
    queued_at: float
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    result: GenerateResult | None = None
    image_url: str | None = None


class JobListResponse(BaseModel):
    jobs: list[JobInfo]
