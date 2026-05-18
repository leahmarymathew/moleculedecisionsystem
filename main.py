"""
FastAPI serving layer for the pharmaceutical analytics pipeline.

Endpoints
---------
POST /analyze           Upload an IQVIA .xlsx file → returns job_id (202)
POST /analyze/json      Submit molecule records as JSON → returns job_id (202)
GET  /jobs/{job_id}     Poll job status / retrieve results
GET  /model/info        Loaded model version and training metadata
GET  /health            Liveness check
GET  /metrics           Prometheus metrics (auto-instrumented)
POST /train             Retrain models and hot-reload (auth-gated)

Run with:
    uvicorn main:app --reload
"""

from __future__ import annotations

import io
import json
import os
import secrets
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import joblib
import pandas as pd
import structlog
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

import metrics as _metrics
from data_layer import IQVIADataLoader
from logging_config import configure_logging, get_logger
from pipeline import run_pipeline
from schemas import (
    AnalysisResult,
    AnalyzeJsonRequest,
    ErrorResponse,
    HealthResponse,
    JobPollResponse,
    JobStatus,
    JobSubmittedResponse,
    ModelInfoResponse,
)
from train_models import train_and_save

# ---------------------------------------------------------------------------
# Logging (configure before any logger is instantiated)
# ---------------------------------------------------------------------------

configure_logging()
_logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODELS_DIR = Path("models/v1")
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB
_UNAUTHENTICATED_PATHS = frozenset({"/health", "/metrics"})

# ---------------------------------------------------------------------------
# In-memory job store
# ---------------------------------------------------------------------------

_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()


def _create_job(request_id: str) -> str:
    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "request_id": request_id,
            "status": JobStatus.queued,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": None,
            "result": None,
            "error": None,
        }
    return job_id


def _get_job(job_id: str) -> Optional[dict]:
    with _jobs_lock:
        return _jobs.get(job_id)


def _update_job(job_id: str, **kwargs: Any) -> None:
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(kwargs)


def _active_job_count() -> int:
    with _jobs_lock:
        return sum(
            1 for j in _jobs.values()
            if j["status"] in (JobStatus.queued, JobStatus.running)
        )


# ---------------------------------------------------------------------------
# Model cache helpers
# ---------------------------------------------------------------------------

def _load_models_cache() -> dict[str, Any]:
    """Load all model artifacts from disk into a fresh dict."""
    cache: dict[str, Any] = {}
    rf_path = MODELS_DIR / "rf_classifier.joblib"
    scaler_path = MODELS_DIR / "rf_scaler.joblib"
    if rf_path.exists() and scaler_path.exists():
        cache["rf_classifier"] = joblib.load(rf_path)
        cache["rf_scaler"] = joblib.load(scaler_path)
    xgb_path = MODELS_DIR / "xgb_model.joblib"
    if xgb_path.exists():
        cache["xgb_model"] = joblib.load(xgb_path)
    kmeans_path = MODELS_DIR / "kmeans.joblib"
    if kmeans_path.exists():
        cache["kmeans"] = joblib.load(kmeans_path)
    return cache


def _reload_models_cache(state: Any) -> None:
    """Hot-reload model artifacts from disk, atomically swapping state.models_cache."""
    new_cache = _load_models_cache()

    meta_path = MODELS_DIR / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            state.model_metadata = json.load(f)

    # Atomic swap — in-flight requests hold references to the old dict (safe).
    state.models_cache = new_cache
    state.models_loaded = bool(new_cache)

    _logger.info("models_cache_reloaded", models_count=len(new_cache))
    _update_model_info_metric(state)


def _update_model_info_metric(state: Any) -> None:
    if state.model_metadata:
        meta = state.model_metadata
        _metrics.model_info.info({
            "trained_at": str(meta.get("trained_at", "")),
            "n_samples": str(meta.get("n_samples", "")),
            "models_dir": state.models_dir,
        })


# ---------------------------------------------------------------------------
# Lifespan: pre-load ML models on startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.models_dir = str(MODELS_DIR)
    app.state.models_loaded = False
    app.state.models_cache: dict[str, Any] = {}
    app.state.model_metadata: Optional[dict] = None

    meta_path = MODELS_DIR / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            app.state.model_metadata = json.load(f)

    app.state.models_cache = _load_models_cache()
    app.state.models_loaded = bool(app.state.models_cache)

    if app.state.models_loaded:
        _logger.info("models_loaded", models_dir=str(MODELS_DIR))
        _update_model_info_metric(app.state)
    else:
        _logger.warning(
            "no_models_found",
            models_dir=str(MODELS_DIR),
            hint="Run train_models.py first",
        )

    yield


# ---------------------------------------------------------------------------
# App + instrumentation
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Pharma Analytics API",
    description="Pharmaceutical molecule investment scoring pipeline",
    version="1.0.0",
    lifespan=lifespan,
)

Instrumentator(
    should_group_status_codes=True,
    should_ignore_untemplated=True,
    excluded_handlers=["/metrics", "/health"],
).instrument(app).expose(app)


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class APIKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        if request.url.path in _UNAUTHENTICATED_PATHS:
            return await call_next(request)
        api_key = os.getenv("API_KEY", "")
        if not api_key:
            return await call_next(request)
        provided = (
            request.headers.get("X-API-Key")
            or request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
        )
        if not secrets.compare_digest(provided, api_key):
            return JSONResponse(
                status_code=401,
                content=ErrorResponse(
                    request_id=structlog.contextvars.get_contextvars().get("request_id", ""),
                    error="Unauthorized",
                    detail="Invalid or missing API key. Provide via X-API-Key header.",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                ).model_dump(),
            )
        return await call_next(request)


# Starlette middleware is LIFO: last added is the outermost wrapper (runs first).
app.add_middleware(APIKeyMiddleware)    # inner — runs second
app.add_middleware(RequestIDMiddleware) # outer — runs first (sets request_id in contextvars)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_request_id() -> str:
    return str(uuid.uuid4())


def _error_response(
    request_id: str,
    status_code: int,
    error: str,
    detail: Optional[str] = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(
            request_id=request_id,
            error=error,
            detail=detail,
            timestamp=datetime.now(timezone.utc).isoformat(),
        ).model_dump(),
    )


def _validate_iqvia_structure(df: pd.DataFrame, request_id: str) -> None:
    """Raise HTTPException(422) if required IQVIA columns are absent."""
    try:
        loader = IQVIADataLoader("")
        loader.raw_df = df
        loader.validate_structure()
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=ErrorResponse(
                request_id=request_id,
                error="Invalid IQVIA structure",
                detail=str(exc),
                timestamp=datetime.now(timezone.utc).isoformat(),
            ).model_dump(),
        )


def _molecules_to_dataframe(body: AnalyzeJsonRequest) -> pd.DataFrame:
    """Convert validated JSON molecules to an IQVIA-compatible DataFrame."""
    return pd.DataFrame([m.model_dump(by_alias=True) for m in body.molecules])


# ---------------------------------------------------------------------------
# Background tasks
# ---------------------------------------------------------------------------

def _run_analysis_task(
    job_id: str,
    df: pd.DataFrame,
    models_cache: dict,
    models_dir: str,
    request_id: str,
) -> None:
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=request_id, job_id=job_id)
    _update_job(job_id, status=JobStatus.running)
    log = _logger.bind(job_id=job_id)
    try:
        log.info("pipeline_started")
        result = run_pipeline(df, models_dir=models_dir, models_cache=models_cache)
        _update_job(
            job_id,
            status=JobStatus.complete,
            completed_at=datetime.now(timezone.utc).isoformat(),
            result={
                "request_id": request_id,
                "job_id": job_id,
                "status": JobStatus.complete,
                "tier_distribution": result["tier_distribution"],
                "ranked_molecules": result["ranked_molecules"],
                "analysis_summary": result["analysis_summary"],
                "validation": result["validation"],
            },
        )
        _metrics.pipeline_runs.labels(status="success").inc()
        for tier, count in result.get("tier_distribution", {}).items():
            _metrics.tier_molecules.labels(tier=tier).inc(count)
        for cls, count in result.get("analysis_summary", {}).get("confidence_levels", {}).items():
            _metrics.confidence_molecules.labels(confidence_class=cls).inc(count)
        log.info("pipeline_complete", tier_distribution=result["tier_distribution"])
    except Exception as exc:
        _metrics.pipeline_runs.labels(status="failure").inc()
        _update_job(
            job_id,
            status=JobStatus.failed,
            completed_at=datetime.now(timezone.utc).isoformat(),
            error=str(exc),
        )
        log.error("pipeline_failed", error=str(exc))


def _retrain_task(job_id: str, data_path: str, request_id: str) -> None:
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=request_id, job_id=job_id)
    _update_job(job_id, status=JobStatus.running)
    log = _logger.bind(job_id=job_id)
    try:
        log.info("model_retrain_started", data_path=data_path)
        train_and_save(data_path, models_dir=str(MODELS_DIR))
        _reload_models_cache(app.state)
        _update_job(
            job_id,
            status=JobStatus.complete,
            completed_at=datetime.now(timezone.utc).isoformat(),
            result={
                "request_id": request_id,
                "job_id": job_id,
                "status": JobStatus.complete,
            },
        )
        log.info("model_retrain_complete")
    except Exception as exc:
        _update_job(
            job_id,
            status=JobStatus.failed,
            completed_at=datetime.now(timezone.utc).isoformat(),
            error=str(exc),
        )
        log.error("model_retrain_failed", error=str(exc))


# ---------------------------------------------------------------------------
# POST /analyze  — multipart .xlsx upload
# ---------------------------------------------------------------------------

@app.post(
    "/analyze",
    response_model=JobSubmittedResponse,
    status_code=202,
    summary="Upload an IQVIA .xlsx file and run the pipeline in the background",
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
    },
)
async def analyze_file(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="IQVIA workbook (.xlsx or .xls)"),
) -> JobSubmittedResponse:
    request_id = _new_request_id()

    if not (file.filename or "").lower().endswith((".xlsx", ".xls")):
        return _error_response(request_id, 400, "Unsupported file type", "Only .xlsx / .xls files are accepted.")

    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        mb = MAX_UPLOAD_BYTES // (1024 * 1024)
        return _error_response(request_id, 413, "File too large", f"Maximum upload size is {mb} MB.")

    try:
        df = pd.read_excel(io.BytesIO(raw), sheet_name="Sheet1")
    except Exception as exc:
        return _error_response(request_id, 422, "Failed to parse Excel file", str(exc))

    _validate_iqvia_structure(df, request_id)

    job_id = _create_job(request_id)
    background_tasks.add_task(
        _run_analysis_task,
        job_id,
        df,
        request.app.state.models_cache,
        request.app.state.models_dir,
        request_id,
    )

    return JobSubmittedResponse(
        job_id=job_id,
        request_id=request_id,
        poll_url=str(request.url_for("poll_job", job_id=job_id)),
    )


# ---------------------------------------------------------------------------
# POST /analyze/json  — programmatic JSON ingestion
# ---------------------------------------------------------------------------

@app.post(
    "/analyze/json",
    response_model=JobSubmittedResponse,
    status_code=202,
    summary="Submit molecule records as JSON and run the pipeline in the background",
    responses={422: {"model": ErrorResponse}},
)
async def analyze_json(
    request: Request,
    background_tasks: BackgroundTasks,
    body: AnalyzeJsonRequest,
) -> JobSubmittedResponse:
    request_id = _new_request_id()

    df = _molecules_to_dataframe(body)
    _validate_iqvia_structure(df, request_id)

    job_id = _create_job(request_id)
    background_tasks.add_task(
        _run_analysis_task,
        job_id,
        df,
        request.app.state.models_cache,
        request.app.state.models_dir,
        request_id,
    )

    return JobSubmittedResponse(
        job_id=job_id,
        request_id=request_id,
        poll_url=str(request.url_for("poll_job", job_id=job_id)),
    )


# ---------------------------------------------------------------------------
# GET /jobs/{job_id}  — poll for result
# ---------------------------------------------------------------------------

@app.get(
    "/jobs/{job_id}",
    response_model=JobPollResponse,
    name="poll_job",
    summary="Poll job status and retrieve results when complete",
    responses={404: {"model": ErrorResponse}},
)
async def poll_job(job_id: str, request: Request) -> JobPollResponse:
    job = _get_job(job_id)

    if job is None:
        raise HTTPException(
            status_code=404,
            detail=ErrorResponse(
                request_id=_new_request_id(),
                error="Job not found",
                detail=f"No job with id={job_id}",
                timestamp=datetime.now(timezone.utc).isoformat(),
            ).model_dump(),
        )

    result: Optional[AnalysisResult] = None
    if job["status"] == JobStatus.complete and job["result"]:
        result = AnalysisResult(**job["result"])
    elif job["status"] == JobStatus.failed:
        result = AnalysisResult(
            request_id=job["request_id"],
            job_id=job_id,
            status=JobStatus.failed,
            error=job.get("error"),
        )

    return JobPollResponse(
        job_id=job_id,
        request_id=job["request_id"],
        status=job["status"],
        created_at=job["created_at"],
        completed_at=job.get("completed_at"),
        result=result,
    )


# ---------------------------------------------------------------------------
# POST /train  — retrain models and hot-reload (auth-gated)
# ---------------------------------------------------------------------------

@app.post(
    "/train",
    response_model=JobSubmittedResponse,
    status_code=202,
    summary="Retrain ML models from source data and hot-reload without restarting",
    responses={
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
    },
)
async def train_models_endpoint(
    request: Request,
    background_tasks: BackgroundTasks,
    data_path: Optional[str] = None,
) -> JobSubmittedResponse:
    request_id = _new_request_id()

    resolved_path = data_path or os.getenv("TRAINING_DATA_PATH", "")
    if not resolved_path:
        return _error_response(
            request_id, 400,
            "Missing data_path",
            "Provide data_path query parameter or set TRAINING_DATA_PATH env var.",
        )

    job_id = _create_job(request_id)
    background_tasks.add_task(_retrain_task, job_id, resolved_path, request_id)

    return JobSubmittedResponse(
        job_id=job_id,
        request_id=request_id,
        poll_url=str(request.url_for("poll_job", job_id=job_id)),
    )


# ---------------------------------------------------------------------------
# GET /model/info
# ---------------------------------------------------------------------------

@app.get(
    "/model/info",
    response_model=ModelInfoResponse,
    summary="Return loaded model version and training metadata",
)
async def model_info(request: Request) -> ModelInfoResponse:
    meta: Optional[dict] = request.app.state.model_metadata

    if meta is None:
        return ModelInfoResponse(
            models_loaded=False,
            models_dir=request.app.state.models_dir,
        )

    return ModelInfoResponse(
        models_loaded=request.app.state.models_loaded,
        models_dir=request.app.state.models_dir,
        trained_at=meta.get("trained_at"),
        n_samples=meta.get("n_samples"),
        rf_holdout_accuracy=(meta.get("rf") or {}).get("holdout_accuracy"),
        xgboost_available=(meta.get("xgboost") or {}).get("available", False),
        xgboost_test_score=(meta.get("xgboost") or {}).get("test_score"),
        n_clusters=(meta.get("kmeans") or {}).get("n_clusters"),
        silhouette_score=(meta.get("kmeans") or {}).get("silhouette_score"),
    )


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness check — returns 'degraded' if models are not loaded",
)
async def health(request: Request) -> HealthResponse:
    return HealthResponse(
        status="ok" if request.app.state.models_loaded else "degraded",
        models_loaded=request.app.state.models_loaded,
        timestamp=datetime.now(timezone.utc).isoformat(),
        active_jobs=_active_job_count(),
    )


# ---------------------------------------------------------------------------
# GET /  — Root endpoint
# ---------------------------------------------------------------------------

@app.get(
    "/",
    response_model=None,
    summary="API root — list available endpoints",
)
async def root():
    return {
        "message": "Pharma Analytics API",
        "version": "1.0.0",
        "docs": "/docs",
        "openapi": "/openapi.json",
        "health": "/health",
        "endpoints": {
            "POST /analyze": "Upload IQVIA .xlsx file",
            "POST /analyze/json": "Submit molecules as JSON",
            "POST /train": "Retrain models and hot-reload",
            "GET /jobs/{job_id}": "Poll job status",
            "GET /model/info": "Get model metadata",
            "GET /health": "Liveness check",
            "GET /metrics": "Prometheus metrics",
        },
    }
