"""Pydantic v2 request/response schemas for the FastAPI serving layer."""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    complete = "complete"
    failed = "failed"


# ---------------------------------------------------------------------------
# IQVIA molecule record — JSON ingestion path
# ---------------------------------------------------------------------------

class MoleculeRecord(BaseModel):
    """One row from an IQVIA workbook expressed as JSON.

    Field aliases match the original Excel column headers so
    ``model_dump(by_alias=True)`` produces a DataFrame-ready dict.
    """
    model_config = ConfigDict(populate_by_name=True)

    # Hierarchy / classification
    Country: str
    Sector: str
    Manufacturer: str
    Molecule_List: str = Field(alias="Molecule List")
    ATC1: Optional[str] = None
    ATC2: Optional[str] = None
    ATC3: Optional[str] = None
    ATC4: Optional[str] = None
    Innovation_Insights: Optional[str] = Field(None, alias="Innovation Insights")

    # Revenue — LCD Manufacturer
    MAT_Q2_2023_LCD: float = Field(0.0, alias="MAT Q2 2023_LCD MNF")
    MAT_Q2_2024_LCD: float = Field(0.0, alias="MAT Q2 2024_LCD MNF")
    MAT_Q2_2025_LCD: float = Field(0.0, alias="MAT Q2 2025_LCD MNF")

    # Volume — Standard Units
    MAT_Q2_2023_SU: float = Field(0.0, alias="MAT Q2 2023_Standard Units")
    MAT_Q2_2024_SU: float = Field(0.0, alias="MAT Q2 2024_Standard Units")
    MAT_Q2_2025_SU: float = Field(0.0, alias="MAT Q2 2025_Standard Units")


class AnalyzeJsonRequest(BaseModel):
    molecules: list[MoleculeRecord] = Field(min_length=1)


# ---------------------------------------------------------------------------
# Job submission response (202 Accepted)
# ---------------------------------------------------------------------------

class JobSubmittedResponse(BaseModel):
    job_id: str
    request_id: str
    status: JobStatus = JobStatus.queued
    poll_url: str


# ---------------------------------------------------------------------------
# Analysis result — embedded once a job completes
# ---------------------------------------------------------------------------

class AnalysisResult(BaseModel):
    request_id: str
    job_id: str
    status: JobStatus
    tier_distribution: Optional[dict[str, int]] = None
    ranked_molecules: Optional[list[dict[str, Any]]] = None
    analysis_summary: Optional[dict[str, Any]] = None
    validation: Optional[dict[str, Any]] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Job poll response
# ---------------------------------------------------------------------------

class JobPollResponse(BaseModel):
    job_id: str
    request_id: str
    status: JobStatus
    created_at: str
    completed_at: Optional[str] = None
    result: Optional[AnalysisResult] = None


# ---------------------------------------------------------------------------
# Model info
# ---------------------------------------------------------------------------

class ModelInfoResponse(BaseModel):
    models_loaded: bool
    models_dir: str
    trained_at: Optional[str] = None
    n_samples: Optional[int] = None
    rf_holdout_accuracy: Optional[float] = None
    xgboost_available: bool = False
    xgboost_test_score: Optional[float] = None
    n_clusters: Optional[int] = None
    silhouette_score: Optional[float] = None


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str          # "ok" | "degraded"
    models_loaded: bool
    timestamp: str
    active_jobs: int


# ---------------------------------------------------------------------------
# Structured error (returned for all 4xx / 5xx)
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    request_id: str
    error: str
    detail: Optional[str] = None
    timestamp: str
