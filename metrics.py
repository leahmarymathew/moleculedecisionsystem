"""Custom Prometheus metrics for the analytics pipeline."""
from prometheus_client import Counter, Info

pipeline_runs = Counter(
    "pharma_pipeline_runs_total",
    "Total number of pipeline runs",
    ["status"],
)

tier_molecules = Counter(
    "pharma_tier_molecules_total",
    "Number of molecules assigned per investment tier",
    ["tier"],
)

confidence_molecules = Counter(
    "pharma_confidence_molecules_total",
    "Number of molecules per confidence class",
    ["confidence_class"],
)

model_info = Info(
    "pharma_model",
    "Metadata for the currently loaded model",
)
