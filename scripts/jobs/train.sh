#!/bin/bash
#SBATCH --job-name=train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=400G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out

# Run a single training job on the cluster.
#
# Submit: sbatch scripts/jobs/train.sh <task>
# Example: sbatch scripts/jobs/train.sh eform

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: sbatch $0 <task>"
    echo "Example: sbatch $0 eform"
    exit 1
fi

TASK="$1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="${SLURM_SUBMIT_DIR:-$SCRIPT_DIR/../..}"
RUN_DIR="$(cd "$RUN_DIR" && pwd)"

if [ ! -f "$RUN_DIR/run.py" ]; then
    echo "train.sh must be submitted from the project root/current run directory." >&2
    echo "Expected run.py at: $RUN_DIR/run.py" >&2
    echo "Run from the repo root, for example:" >&2
    echo "  cd /path/to/mmct && sbatch scripts/jobs/train.sh $TASK" >&2
    exit 1
fi

if [[ ! "$TASK" =~ ^[a-zA-Z0-9_]+$ ]] || [ ! -f "$RUN_DIR/config/task/$TASK.yaml" ]; then
    echo "Unknown task: $TASK (expected a config/task/<task>.yaml file)." >&2
    exit 1
fi

if [ ! -x "$RUN_DIR/.venv/bin/python" ]; then
    echo "Python environment not found at: $RUN_DIR/.venv/bin/python" >&2
    exit 1
fi

if [ ! -f "$RUN_DIR/data/element_properties/mat2vec.csv" ]; then
    echo "Element embeddings not found at: $RUN_DIR/data/element_properties/mat2vec.csv" >&2
    exit 1
fi

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

PROJECT_NAME="$(basename "$RUN_DIR")"
JOB_TAG="${SLURM_JOB_ID:-manual_$$}"
RESULTS_DIR="$SCRATCH_BASE/$PROJECT_NAME/runs/${TASK}_${JOB_TAG}"

mkdir -p "$RESULTS_DIR"
RESULTS_DIR="$(cd "$RESULTS_DIR" && pwd)"
# CrabNet reads element embeddings relative to its working directory.
if [ ! -e "$RESULTS_DIR/data" ]; then
    ln -s "$RUN_DIR/data" "$RESULTS_DIR/data"
fi

cd "$RUN_DIR"

export PYTHONPATH="$RUN_DIR${PYTHONPATH:+:$PYTHONPATH}"
export HYDRA_FULL_ERROR=1

echo "Task: $TASK"
echo "Project: $RUN_DIR"
echo "Results: $RESULTS_DIR"

exec "$RUN_DIR/.venv/bin/python" -u "$RUN_DIR/run.py" \
    task="$TASK" \
    "hydra.run.dir=$RESULTS_DIR" \
    hydra.job.chdir=true
