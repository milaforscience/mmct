#!/bin/bash
# Submit one MMCT training job for each of the five benchmark tasks.
#
# Usage: bash scripts/jobs/run_all_tasks.sh
# Example: bash scripts/jobs/run_all_tasks.sh

set -euo pipefail

if [ "$#" -ne 0 ]; then
    echo "Usage: bash $0" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [ ! -f "$PROJECT_ROOT/run.py" ] || [ ! -x "$PROJECT_ROOT/.venv/bin/python" ]; then
    echo "Expected run.py and .venv/bin/python in: $PROJECT_ROOT" >&2
    exit 1
fi

if ! command -v sbatch >/dev/null 2>&1; then
    echo "sbatch is required to submit training jobs." >&2
    exit 1
fi

TASKS=(eform bandgap dielectric log_kvrh log_gvrh)
for TASK in "${TASKS[@]}"; do
    if [ ! -f "$PROJECT_ROOT/config/task/$TASK.yaml" ]; then
        echo "Task configuration not found: config/task/$TASK.yaml" >&2
        exit 1
    fi
done

USER_NAME="${USER:-$(whoami)}"
SCRATCH_BASE=""

for CANDIDATE in "${SCRATCH:-}" "/network/scratch/${USER_NAME:0:1}/$USER_NAME" "$HOME/scratch"; do
    if [ -n "$CANDIDATE" ] && mkdir -p "$CANDIDATE" 2>/dev/null; then
        SCRATCH_BASE="$(cd "$CANDIDATE" && pwd)"
        break
    fi
done

if [ -z "$SCRATCH_BASE" ]; then
    echo "Could not find or create a scratch directory." >&2
    exit 1
fi

PROJECT_NAME="$(basename "$PROJECT_ROOT")"
SCRATCH_DIR="$SCRATCH_BASE/$PROJECT_NAME"
LOGS_DIR="$SCRATCH_DIR/logs"
cd "$PROJECT_ROOT"
mkdir -p "$LOGS_DIR"

echo "========================================"
echo " MMCT task submission"
echo "========================================"
echo ""

for TASK in "${TASKS[@]}"; do
    JID=$(sbatch --parsable \
        --chdir="$PROJECT_ROOT" \
        --job-name="train_${TASK}" \
        --output="$LOGS_DIR/train_${TASK}_%j.out" \
        --error="$LOGS_DIR/train_${TASK}_%j.out" \
        "$SCRIPT_DIR/train.sh" "$TASK")
    echo "  train  $TASK  → job $JID"
done

echo "========================================"
echo "All jobs submitted."
echo "Monitor with: squeue -u $USER_NAME"
echo "Logs in: $LOGS_DIR"
echo "========================================"
