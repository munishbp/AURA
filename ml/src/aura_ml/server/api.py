"""REST API around AuraInferencePipeline.

Endpoints (all under /v1):

    GET  /v1/health              status, GPU, what's loaded
    GET  /v1/procedures          supported procedures + example instructions
    POST /v1/expand              photo + instruction -> expanded prompt (fast, ~3 s)
    POST /v1/generate            photo + instruction -> edited image (sync, ~1-2 min)
    POST /v1/jobs/generate       same, async: returns a job id immediately
    GET  /v1/jobs                recent jobs
    GET  /v1/jobs/{id}           job status + result metadata
    GET  /v1/jobs/{id}/image     the generated PNG
    POST /v1/metrics             score a (source, output, instruction) triple

Design notes:
- One pipeline instance per process. All diffusion work funnels through a
  single-worker executor + the pipeline's gpu_lock, so concurrent requests
  queue instead of OOMing the 5090. /expand shares the lock-free path (the
  expander is a separate model and a 3 s call).
- Sync /generate is intentionally supported (curl-friendly); anything
  building a UI should prefer the job endpoints.
- With --ui, the Gradio demo is mounted at /ui sharing the SAME pipeline.
"""

from __future__ import annotations

import base64
import io
import math
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image

from aura_ml.inference.pipeline import (
    PROCEDURES,
    AuraInferencePipeline,
    build_default_pipeline,
)
from aura_ml.prompt_expander.qwen35 import (
    EXAMPLE_INSTRUCTIONS,
    PROCEDURES as PROCEDURE_SPECS,
    OutOfScopeError,
)
from aura_ml.server.schemas import (
    ExpandResponse,
    GenerateResponse,
    GenerateResult,
    HealthResponse,
    JobInfo,
    JobListResponse,
    JobSubmitResponse,
    MetricsResponse,
    ProcedureInfo,
)

try:
    from aura_ml.eval.metrics import all_metrics, is_static

    _HAVE_METRICS = True
except ImportError:
    _HAVE_METRICS = False

MAX_UPLOAD_BYTES = 30 * 1024 * 1024
MAX_JOBS_KEPT = 100


@dataclass
class ServerConfig:
    checkpoints_dir: str = "checkpoints"
    use_prompt_expander: bool = True
    preload: bool = False  # load both models at startup instead of first request
    mount_ui: bool = False  # serve the Gradio demo at /ui


@dataclass
class Job:
    job_id: str
    status: str = "queued"  # queued | running | done | error
    queued_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    result: GenerateResult | None = None
    image_png: bytes | None = None


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def add(self, job: Job) -> None:
        with self._lock:
            self._jobs[job.job_id] = job
            # Evict oldest finished jobs beyond the cap.
            if len(self._jobs) > MAX_JOBS_KEPT:
                finished = sorted(
                    (j for j in self._jobs.values() if j.status in ("done", "error")),
                    key=lambda j: j.queued_at,
                )
                for j in finished[: len(self._jobs) - MAX_JOBS_KEPT]:
                    self._jobs.pop(j.job_id, None)

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def recent(self, n: int = 20) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.queued_at, reverse=True)[:n]


def _read_upload(file: UploadFile) -> Image.Image:
    raw = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "image larger than 30 MB")
    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"not a decodable image: {e}") from e


def _check_procedure(procedure: str) -> str:
    if procedure not in PROCEDURES:
        raise HTTPException(
            422, f"unknown procedure '{procedure}' — one of {list(PROCEDURES)}"
        )
    return procedure


def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _nan_to_none(v: float) -> float | None:
    return None if (isinstance(v, float) and math.isnan(v)) else v


def _compute_metrics(source: Image.Image, output: Image.Image, prompt: str) -> MetricsResponse:
    m = all_metrics(source, output, prompt)
    return MetricsResponse(
        edit_magnitude=_nan_to_none(m["edit_magnitude"]),
        arcface_cosine=_nan_to_none(m["arcface_cosine"]),
        lpips=_nan_to_none(m["lpips"]),
        clip_score=_nan_to_none(m["clip_score"]),
        canary_static=is_static(m),
    )


def create_app(config: ServerConfig | None = None) -> FastAPI:
    config = config or ServerConfig()
    pipeline: AuraInferencePipeline = build_default_pipeline(
        Path(config.checkpoints_dir), use_prompt_expander=config.use_prompt_expander
    )
    # ONE worker: the GPU can run exactly one diffusion at a time.
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="aura-gpu")
    jobs = JobStore()

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _lifespan(_: FastAPI):
        if config.preload:
            executor.submit(pipeline.diffuser.load)
            if pipeline.expander is not None:
                executor.submit(pipeline.expander.load)
        yield
        executor.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(
        title="Aura API",
        version="1.0",
        description="AI-assisted plastic-surgery outcome visualization. "
        "Photo + instruction in, realistic post-surgical preview out. "
        "For physician consultation and research purposes only — not a medical device.",
        lifespan=_lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @app.get("/v1/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        import torch

        cuda = torch.cuda.is_available()
        return HealthResponse(
            cuda_available=cuda,
            gpu_name=torch.cuda.get_device_name(0) if cuda else None,
            vram_used_gb=round(torch.cuda.memory_allocated() / 2**30, 2) if cuda else None,
            vram_total_gb=round(
                torch.cuda.get_device_properties(0).total_memory / 2**30, 2
            ) if cuda else None,
            editor_loaded=pipeline.diffuser._pipe is not None,
            expander_enabled=pipeline.expander is not None,
            expander_loaded=pipeline.expander is not None
            and pipeline.expander._model is not None,
            loras_loaded=sorted(pipeline.diffuser._loaded_loras),
            metrics_available=_HAVE_METRICS,
        )

    @app.get("/v1/procedures", response_model=list[ProcedureInfo])
    def procedures() -> list[ProcedureInfo]:
        return [
            ProcedureInfo(
                name=name,
                description=PROCEDURE_SPECS[name].context,
                example_instruction=EXAMPLE_INSTRUCTIONS[name],
            )
            for name in PROCEDURES
        ]

    # ------------------------------------------------------------------
    # Expansion
    # ------------------------------------------------------------------

    @app.post("/v1/expand", response_model=ExpandResponse)
    def expand(
        image: UploadFile = File(..., description="face photo (jpeg/png/webp)"),
        instruction: str = Form(..., min_length=1),
        procedure: str = Form(...),
        seed: int | None = Form(None),
    ) -> ExpandResponse:
        _check_procedure(procedure)
        img = _read_upload(image)
        try:
            result = pipeline.expand_prompt(img, instruction, procedure, seed=seed)
        except OutOfScopeError as e:
            raise HTTPException(
                422, {"error": "out_of_scope", "procedure": procedure, "reason": str(e)}
            ) from e
        return ExpandResponse(
            prompt=result.prompt,
            procedure=result.procedure,
            used_fallback=result.used_fallback,
            latency_s=round(result.latency_s, 2),
        )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _run_generation(
        img: Image.Image,
        instruction: str,
        procedure: str,
        prompt_override: str | None,
        num_steps: int | None,
        seed: int | None,
        use_expander: bool,
        return_metrics: bool,
    ) -> tuple[GenerateResult, bytes]:
        """Executed on the single GPU worker thread."""
        t0 = time.monotonic()
        with pipeline.gpu_lock:
            if prompt_override:
                prompt_used = prompt_override.strip()
                expander_used = False
            elif use_expander:
                expansion = pipeline.expand_prompt(img, instruction, procedure, seed=seed)
                prompt_used = expansion.prompt
                expander_used = not expansion.used_fallback
            else:
                prompt_used = instruction.strip()
                expander_used = False

            edited, _ = pipeline.generate(
                img, prompt_used, procedure, num_steps=num_steps, seed=seed, expand=False
            )

        metrics = None
        if return_metrics and _HAVE_METRICS:
            metrics = _compute_metrics(img, edited, prompt_used)

        result = GenerateResult(
            prompt_used=prompt_used,
            procedure=procedure,
            seed=seed,
            num_steps=num_steps or pipeline.diffuser.default_num_steps,
            expander_used=expander_used,
            latency_s=round(time.monotonic() - t0, 2),
            metrics=metrics,
        )
        return result, _png_bytes(edited)

    def _parse_generate_form(
        image: UploadFile,
        instruction: str | None,
        procedure: str,
        prompt_override: str | None,
        num_steps: int | None,
        seed: int | None,
    ):
        _check_procedure(procedure)
        if not (instruction and instruction.strip()) and not (
            prompt_override and prompt_override.strip()
        ):
            raise HTTPException(422, "provide `instruction` or `prompt_override`")
        if num_steps is not None and not 4 <= num_steps <= 80:
            raise HTTPException(422, "num_steps must be in [4, 80]")
        return _read_upload(image)

    @app.post(
        "/v1/generate",
        response_model=GenerateResponse,
        responses={200: {"content": {"image/png": {}}}},
    )
    def generate(
        image: UploadFile = File(...),
        instruction: str | None = Form(None),
        procedure: str = Form(...),
        prompt_override: str | None = Form(
            None, description="skip expansion and use this prompt verbatim"
        ),
        num_steps: int | None = Form(None, description="default: the active recipe (8 with Lightning, 40 without)"),
        seed: int | None = Form(None),
        use_expander: bool = Form(True),
        return_metrics: bool = Form(True),
        response_format: str = Form("json", pattern="^(json|image)$"),
    ):
        """Synchronous generation. Expect ~1-2 minutes on the 5090; requests
        are queued one at a time. UIs should use POST /v1/jobs/generate."""
        img = _parse_generate_form(
            image, instruction, procedure, prompt_override, num_steps, seed
        )
        try:
            result, png = executor.submit(
                _run_generation,
                img, instruction or "", procedure, prompt_override,
                num_steps, seed, use_expander, return_metrics,
            ).result()
        except OutOfScopeError as e:
            raise HTTPException(
                422, {"error": "out_of_scope", "procedure": procedure, "reason": str(e)}
            ) from e

        if response_format == "image":
            return Response(content=png, media_type="image/png")
        return GenerateResponse(
            **result.model_dump(), image_b64=base64.b64encode(png).decode()
        )

    # ------------------------------------------------------------------
    # Async jobs
    # ------------------------------------------------------------------

    def _job_info(job: Job) -> JobInfo:
        return JobInfo(
            job_id=job.job_id,
            status=job.status,
            queued_at=job.queued_at,
            started_at=job.started_at,
            finished_at=job.finished_at,
            error=job.error,
            result=job.result,
            image_url=f"/v1/jobs/{job.job_id}/image" if job.image_png else None,
        )

    @app.post("/v1/jobs/generate", response_model=JobSubmitResponse, status_code=202)
    def submit_job(
        image: UploadFile = File(...),
        instruction: str | None = Form(None),
        procedure: str = Form(...),
        prompt_override: str | None = Form(None),
        num_steps: int | None = Form(None, description="default: the active recipe (8 with Lightning, 40 without)"),
        seed: int | None = Form(None),
        use_expander: bool = Form(True),
        return_metrics: bool = Form(True),
    ) -> JobSubmitResponse:
        img = _parse_generate_form(
            image, instruction, procedure, prompt_override, num_steps, seed
        )
        job = Job(job_id=uuid.uuid4().hex[:12])
        jobs.add(job)

        def _worker() -> None:
            job.status = "running"
            job.started_at = time.time()
            try:
                job.result, job.image_png = _run_generation(
                    img, instruction or "", procedure, prompt_override,
                    num_steps, seed, use_expander, return_metrics,
                )
                job.status = "done"
            except Exception as e:  # surfaced via GET /v1/jobs/{id}
                job.status = "error"
                job.error = f"{type(e).__name__}: {e}"
            finally:
                job.finished_at = time.time()

        executor.submit(_worker)
        return JobSubmitResponse(
            job_id=job.job_id,
            status=job.status,
            status_url=f"/v1/jobs/{job.job_id}",
            image_url=f"/v1/jobs/{job.job_id}/image",
        )

    @app.get("/v1/jobs", response_model=JobListResponse)
    def list_jobs() -> JobListResponse:
        return JobListResponse(jobs=[_job_info(j) for j in jobs.recent()])

    @app.get("/v1/jobs/{job_id}", response_model=JobInfo)
    def job_status(job_id: str) -> JobInfo:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "no such job")
        return _job_info(job)

    @app.get("/v1/jobs/{job_id}/image")
    def job_image(job_id: str) -> Response:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "no such job")
        if job.status == "error":
            raise HTTPException(409, f"job failed: {job.error}")
        if job.image_png is None:
            raise HTTPException(409, f"job not finished (status={job.status})")
        return Response(content=job.image_png, media_type="image/png")

    # ------------------------------------------------------------------
    # Standalone metrics
    # ------------------------------------------------------------------

    @app.post("/v1/metrics", response_model=MetricsResponse)
    def metrics(
        source: UploadFile = File(..., description="the 'before' photo"),
        output: UploadFile = File(..., description="the edited photo"),
        instruction: str = Form(...),
    ) -> MetricsResponse:
        if not _HAVE_METRICS:
            raise HTTPException(
                501, "eval metrics not installed — `uv sync --extra eval`"
            )
        return _compute_metrics(_read_upload(source), _read_upload(output), instruction)

    # ------------------------------------------------------------------
    # Optional Gradio UI on the same pipeline
    # ------------------------------------------------------------------

    if config.mount_ui:
        try:
            import gradio as gr

            from app.demo import build_ui  # resolvable when run from ml/

            gr.mount_gradio_app(app, build_ui(pipeline), path="/ui")
            print("[server] Gradio UI mounted at /ui (same pipeline as the API)")
        except ImportError as e:
            print(f"[server] --ui requested but UI not mountable: {e}")

    return app
