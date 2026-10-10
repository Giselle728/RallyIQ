#!/usr/bin/env bash
# ==============================================================================
# RallyIQ Big Data Pipeline Runner
# Ingests, validates, and runs PySpark distributed aggregations.
# ==============================================================================

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

echo "======================================================================"
echo " RallyIQ: Big Data Tennis Intelligence Platform Pipeline"
echo "======================================================================"

VENV_PYTHON="$PROJECT_DIR/venv/bin/python3"

export PYSPARK_PYTHON="$VENV_PYTHON"
export PYSPARK_DRIVER_PYTHON="$VENV_PYTHON"

echo "[1/3] Running Real Dataset Ingestion, Validation & Profiling..."
"$VENV_PYTHON" -m backend.pipeline.ingest_validate

echo "[2/3] Running PySpark Distributed Batch Transformations..."
"$VENV_PYTHON" -m backend.pipeline.spark_processor

echo "[3/3] Generating Replay & Streaming Indexes..."
"$VENV_PYTHON" -m backend.api.replay_indexer

echo "======================================================================"
echo " ✓ RallyIQ Pipeline successfully executed."
echo "======================================================================"
