#!/bin/bash
# Submit one MMCT training job for each of the five benchmark tasks.
#
# Usage: bash scripts/jobs/run_all_tasks.sh
# Example: bash scripts/jobs/run_all_tasks.sh
# Run jobs in the project directory, using the normal local output folders.
# Slurm output and error logs are also written to the project directory.

set -euo pipefail

if [ "$#" -ne 0 ]; then
    echo "Usage: bash $0" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOGS_DIR="$PROJECT_ROOT"

if [ ! -f "$PROJECT_ROOT/run.py" ] || [ ! -x "$PROJECT_ROOT/.venv/bin/python" ]; then
    echo "Expected run.py and .venv/bin/python in: $PROJECT_ROOT" >&2
    exit 1
fi

if ! command -v sbatch >/dev/null 2>&1; then
    echo "sbatch is required to submit training jobs." >&2
    exit 1
fi

TASKS=(eform bandgap dielectric log_kvrh log_gvrh phonons)
for TASK in "${TASKS[@]}"; do
    if [ ! -f "$PROJECT_ROOT/config/task/$TASK.yaml" ]; then
        echo "Task configuration not found: config/task/$TASK.yaml" >&2
        exit 1
    fi
done

USER_NAME="${USER:-$(whoami)}"
cd "$PROJECT_ROOT"

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
echo "Results in: $PROJECT_ROOT"
echo "========================================"
