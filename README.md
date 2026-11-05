# Bat Algorithm Clustering and Parameter Sensitivity

Research project: **Application of Bat Algorithm to Data Clustering and Parameter
Sensitivity Analysis**. Optimise K centroids with the specified Standard Continuous
Bat Algorithm (BA), minimising total Sum of Squared Errors (SSE), and investigate
`A0`, `r0`, `alpha`, `gamma`, `f_min`, and `f_max`.

## Current state

Implemented and tested: continuous BA, pure SSE, centroid assignment and empty-cluster
reporting, multi-start Lloyd K-Means, uniform population initialization, z-score
preprocessing, baseline/OFAT/grid experiments, shared-seed repetitions, JSONL/CSV export,
resume protection, descriptive statistics, paired Wilcoxon/Holm comparisons, and PNG plots.
Historical pilot files exist; they are exploratory evidence. Formal discovery and
confirmation have not been run in this takeover. See [handoff](docs/HANDOFF.md).

## Installation

Run commands from `ba-clustering/`. Python >=3.11 is declared; Python 3.12.13 was verified.
The existing `.venv` already contains the dependencies and editable package. To recreate:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install "setuptools>=68" wheel
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip install --no-deps --no-build-isolation -e .
```

Fresh dependency installation remains unverified. An isolated offline attempt failed
because the build backend was missing; the setup recipe above explicitly supplies it.
See [reproducibility](docs/REPRODUCIBILITY.md) for the exact verified environment and limits. Prefix CLI commands with `.venv/bin/` when not activating the environment.

## Quick start and CLI

```bash
.venv/bin/ba-clustering --help
.venv/bin/ba-clustering list
.venv/bin/ba-clustering run --dataset iris --seed 0 --set max_iterations=5
```

`run` prints JSON and writes no results. Dataset names come from `configs/default.yaml`.
`--set NAME=VALUE` may be repeated for BA settings. `--config PATH` selects a base file.

Use a fresh output root for verification (the directory returned by `mktemp` is unique):

```bash
BA_SMOKE_ROOT=$(mktemp -d /tmp/ba-smoke.XXXXXX)
.venv/bin/ba-clustering experiment smoke --output "$BA_SMOKE_ROOT"
.venv/bin/ba-clustering experiment smoke --output "$BA_SMOKE_ROOT" --resume
.venv/bin/ba-clustering plot "$BA_SMOKE_ROOT/smoke"
```

Experiments use `configs/experiments.yaml`; `--experiments PATH` selects another file.
Existing results cause an error unless `--resume` or `--overwrite` is explicit. Overwrite
is destructive: prefer a new root. Plot generation regenerates derived PNGs.
The scripts `run_baseline.py`, `run_parameter_sweep.py`, and `generate_plots.py` wrap the
CLI; their no-argument defaults run pilots, so inspect the budget before using them.
Do not run `ofat_discovery` or `confirmation` until the protocol and candidate are frozen.

## Python API

```python
import numpy as np
from ba_clustering import BAConfig, BatClustering, compute_sse

X = np.array([[1.0], [2.0], [8.0], [9.0]])
config = BAConfig(population_size=5, max_iterations=3, n_clusters=2,
                  A0=0.5, r0=0.5, alpha=0.9, gamma=0.9,
                  f_min=0.0, f_max=2.0, seed=0)
model = BatClustering(config).fit(X)
assert model.best_fitness_ == compute_sse(X, model.centroids_)
print(model.centroids_, model.predict(X))
```

The estimator does not scale inputs automatically; use `preprocess` when desired.
External class labels are used only for ARI evaluation, never optimisation.

## Verification

```bash
.venv/bin/python -m pytest
.venv/bin/python -m pytest -m integration
.venv/bin/ruff check src tests scripts
.venv/bin/ruff format --check src tests scripts
.venv/bin/mypy
```

For a small instrumented investigation on smoke seeds 0 and 1 only:

```bash
.venv/bin/python scripts/audit_frequency.py --output /tmp/ba-frequency-new.json
```

Choose an unused output file: diagnostics refuse to replace it.

## Outputs and interpretation

Each experiment writes `<output>/<id>/manifest.json`, `runs.jsonl`, `runs.csv`,
`summary.csv`, and `paired_vs_reference.csv`; plots go to `figures/` on request.
JSONL includes convergence histories and acceptance counts. Manifests snapshot the
configuration, preprocessing, data hash and environment. Resume compares definition/data
hashes, but does not reject software/environment changes; avoid mixing versions.

The velocity is deliberately `v + (position - best) * frequency`. Global moves usually
fail to improve in current pilots, so local search dominates and frequency effects are
weak under these settings. This is not a universal result or evidence of superiority to
K-Means. No hybrid refinement, alternative velocity sign, empty-cluster repair, feature
selection, or other optimisation algorithm is included. Local step size depends on feature
scale. OFAT misses interactions, and confidence intervals on ten pilot seeds are limited.

## Documentation

Start with [INDEX](docs/INDEX.md), [algorithm specification](docs/ALGORITHM_SPEC.md),
[API](docs/API_REFERENCE.md), [data contract](docs/DATA_CONTRACT.md),
[experiment design](docs/EXPERIMENT_DESIGN.md), [results guide](docs/RESULTS_GUIDE.md),
[testing](docs/TESTING.md), and [reproducibility](docs/REPRODUCIBILITY.md).
Future agents must read [AGENTS](AGENTS.md) and [HANDOFF](docs/HANDOFF.md) first.


## Notebook và dataset riêng

Mở [parameter_sensitivity.ipynb](notebooks/parameter_sensitivity.ipynb), sửa
`DATASET_PATH`, `N_CLUSTERS`, `LABEL_COLUMN` rồi Run All. Thư mục [datasets](datasets/README.md)
có CSV mẫu; hỗ trợ cả đường dẫn tuyệt đối đến CSV riêng. Notebook so sánh BA/K-Means,
OFAT sáu tham số và grid A0/r0, lưu raw results và 11 biểu đồ/bảng từ lần chạy đã kiểm tra.
Mặc định chỉ dùng seed kỹ thuật 0,1. Xem [hướng dẫn](docs/NOTEBOOK_GUIDE.md).
Toàn bộ cell đã chạy bằng Python runner; kernel VS Code/Jupyter chưa được kiểm chứng
vì tải dependency bị lỗi DNS. Nhóm dependency tùy chọn `.[notebooks]` được khai báo;
không cần nhóm này để dùng `scripts/run_notebook.py`.


Publication scope: the original workspace retains all historical raw pilot and notebook
session directories. The GitHub publication snapshot excludes those bulk generated
results and all private/user datasets; it includes the executed synthetic notebook and
selected audit evidence under docs/verification. A fresh clone will not contain the
historical raw pilot files. Reproduce engineering checks into fresh output directories;
do not present newly generated runs as the original historical evidence.
