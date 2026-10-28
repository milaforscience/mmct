# MMCT

MMCT trains a CrabNet model to predict material properties from chemical
compositions. This repository is a work in progress.

## Requirements and setup

- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Python 3.14 or later (the project declares `requires-python = ">=3.14"`)
- A PyTorch build appropriate for your machine. The default `torch` dependency
  is included in the project lockfile. For CUDA or ROCm, choose a compatible
  build using the [PyTorch installation selector](https://pytorch.org/get-started/locally/).

From the repository root, install the Python version and dependencies:

```bash
uv python install 3.14
uv sync
```

`uv sync` creates `.venv` and installs the locked application and development
dependencies, including pytest. To check whether PyTorch can see a CUDA device:

```bash
uv run python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

## Data required to train

All task configs currently use `/network/projects/crystalgfn/data/benchmark`
as their default data root. This path is on the Mila cluster and is not expected
to be available on other machines. Override `task.root` and `task.src` for your
environment; `task.src` must resolve to the selected task's data directory.

The preferred format stores the cross-validation assignment in a `Fold` column
instead of encoding the fold in filenames. Put the split datasets in:

```text
<task-data>/train_data.csv
<task-data>/val_data.csv
<task-data>/test_data.csv
```

Each file should include `Fold` values from `0` to `n_splits - 1`, plus
`formula` and `target` columns. This is the preferred schema. For compatibility,
the current loader also accepts `Formulae` plus the task target column (for
example, `Eform` for `eform`) and renames them to `formula` and `target`. This
automatic column renaming is also planned for deprecation in a subsequent
version, so new datasets should use the normalized names directly.

Supply `val_data.csv` to use the validation split defined in the data. If it is
absent, the code splits `train_data.csv` using `val_split_ratio` and still
requires its `Fold` column. When `test=true`, test prediction reads
`test_data.csv`.

The alternative format puts the fold index in each filename, for example:

```text
<task-data>/train0.csv  <task-data>/val0.csv  <task-data>/test0.csv
<task-data>/train1.csv  <task-data>/val1.csv  <task-data>/test1.csv
...
```

These files should contain `formula` and `target` columns (or `cif_id` in place
of `formula`). This compatibility format is retained for now to run datasets
from the original CrabNet or upstream repository for comparison and
benchmarking. It is planned for deprecation in a subsequent version; use the
`Fold`-column format for new datasets.

The element-property tables used by the model are in
`mmct/data/element_properties/`; in particular, `mat2vec.csv` is required.

## Run training

Run from the repository root. The working directory is set to `mmct/` so the
model can find the element-property tables at `data/element_properties/`.
Hydra configuration files are loaded from the repository's `config/` folder.

For example, to train the default eform task using the configured benchmark
path:

```bash
uv run --directory mmct python ../run.py \
  task=eform model=mmct_base hydra.job.chdir=false
```

To use a local data directory, override the task paths. This example expects
the CSVs under `/datasets/benchmark/matbench_mp_e_form/`:

```bash
uv run --directory mmct python ../run.py \
  task=eform \
  model=mmct_base \
  task.root=/datasets/benchmark \
  'task.src=$root/matbench_mp_e_form' \
  hydra.job.chdir=false
```

The entry point uses Hydra, so options are passed as `key=value` overrides
rather than positional arguments. To inspect available config choices and
resolved defaults:

```bash
uv run --directory mmct python ../run.py --help
uv run --directory mmct python ../run.py --cfg job
```

Available task configs are `eform`, `bandgap`, `dielectric`, `log_kvrh`,
`log_gvrh`, `perovskites`, and `phonons`. The available model config is
`mmct_base`.

Common overrides include:

| Option | Default | Purpose |
| --- | --- | --- |
| `task=<name>` | `eform` | Select a file under `config/task/`. |
| `model=<name>` | `mmct_base` | Select a file under `config/model/`. |
| `n_splits=<integer>` | `5` | Number of cross-validation folds. |
| `test=true|false` | `true` | Run test-set predictions and write the mean and standard deviation. Validation predictions are run for each fold either way. |
| `seed=<integer>` | `12345` | Set the NumPy and PyTorch seeds. |
| `val_split_ratio=<float>` | `0.2` | Validation fraction when splitting `train_data.csv`. |
| `model.optim.epochs=<integer>` | `300` | Training epochs. |
| `model.optim.losscurve=true|false` | `true` | Display loss curves during training. |
| `task.root=<path>` | cluster path | Benchmark data root. |
| `task.src=<path>` | `$root/matbench_mp_e_form` | Task data directory relative to the root, or an explicit path. |

For a shorter local run, for example:

```bash
uv run --directory mmct python ../run.py \
  task=eform model=mmct_base \
  task.root=/datasets/benchmark \
  'task.src=$root/matbench_mp_e_form' \
  n_splits=2 test=false model.optim.epochs=5 \
  model.optim.losscurve=false hydra.job.chdir=false
```

The model saves checkpoints under `mmct/models/trained_models/`. Prediction
CSVs and training figures are written under the `mmct/` working directory; the
test summary is written next to `run.py`. Create the checkpoint directory
before training if it does not exist:

```bash
mkdir -p mmct/models/trained_models
```

## Tests

Run the utility and model tests plus the `run.py` orchestration integration
test from the repository root:

```bash
uv run pytest -q
```

The integration test stubs model training and prediction while checking
cross-validation dispatch and test-summary output. The tests use small
synthetic inputs and do not need benchmark datasets or a GPU.

## Slurm jobs

The repository includes `scripts/jobs/train.sh` for one task and
`scripts/jobs/run_all_tasks.sh` for the configured task set. They require Slurm
(`sbatch`) and the project `.venv`. The single-task wrapper also checks for
`data/element_properties/mat2vec.csv` at the repository root; the bundled copy
is currently under `mmct/data/element_properties/`, so copy or link that data
directory to the path the wrapper expects before using these scripts.
