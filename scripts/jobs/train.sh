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
# Run from the project directory and save results in its normal output folders.

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

RESULTS_DIR="$RUN_DIR"

cd "$RUN_DIR"

export PYTHONPATH="$RUN_DIR${PYTHONPATH:+:$PYTHONPATH}"
export HYDRA_FULL_ERROR=1

echo "Task: $TASK"
echo "Project: $RUN_DIR"
echo "Results: $RESULTS_DIR"

exec "$RUN_DIR/.venv/bin/python" -u "$RUN_DIR/run.py" \
    task="$TASK" \
    "hydra.run.dir=$RESULTS_DIR" \
    hydra.job.chdir=false
