# Reproducibility

Verified 2026-11-05 on Linux x86_64, Python 3.12.13, package 0.1.0, NumPy 2.5.4,
SciPy 1.18.1, scikit-learn 1.9.1, pandas 3.1.1, Matplotlib 3.11.2, PyYAML 6.0.3,
pytest 9.2.0, Ruff 0.17.3 and mypy 2.4.1. requirements-lock.txt is the existing pinned
runtime/dev environment snapshot. pyproject declares Python>=3.11; only 3.12 was checked.
No usable Git repository, commit or tracked/untracked distinction was available.

## Environment and installation status

The existing .venv and editable package work. Invocations from project root:

```bash
.venv/bin/python -m pytest
.venv/bin/ba-clustering --help
.venv/bin/ba-clustering list
.venv/bin/ba-clustering run --dataset iris --seed 0 --set max_iterations=5
```

A fresh isolated offline editable install with --no-deps --no-build-isolation failed:
setuptools.build_meta unavailable. An attempt to use uv's default cache also failed
because its lock location is read-only. These are setup limits, not test failures.
A complete fresh dependency download/install was not executed. requirements-lock.txt
does not pin the setuptools build backend or provide wheel artifacts. Fresh installation
must supply setuptools>=68 and dependencies declared in pyproject; do not claim an offline
reproduction bundle exists. The README's new-environment recipe is identified as unverified.
No dependency was changed in the working environment.

## Data and configuration provenance

Iris/Wine come from scikit-learn's bundled datasets, no network. Dataset.source records
loader provenance. Synthetic options/seed are recorded in generator metadata; defaults:
300 samples, 4 features, 4 centers, cluster_std=1.5, data seed 42. Raw class labels are
separate and never reach optimization. Do not confuse classes with an SSE optimum.
Z-score uses ddof=0, constant columns mapped to zero, shared by both algorithms.
CSV input is local numeric data with a header; optional external label column is removed.

Manifests snapshot resolved spec, factors, seeds, dataset metadata and processed matrix
SHA256 (X.tobytes), preprocessing mean/scale, software environment and status. Data hash
excludes external labels, so changed evaluation labels are not caught by resume. The spec
includes dataset loader options and data hashes; factor order is retained for iteration,
while canonical JSON hash sorts keys. Changed code with same package version is not detected.
No final centroid/label arrays are saved in run records; reruns reconstruct them.

## Seed and random-draw policy

Use local default_rng per fit. Seed rosters: smoke 0,1; pilot 9000..9009;
discovery 1000..1009; confirmation 2000..2019. Shipped rosters are disjoint and tested.
Initialization draws first, then each iteration draws beta, pulse, epsilon, loudness,
including unused draws. Keep draw order, distributions, shape and consumption fixed.
Pair algorithm seeds across conditions; population size, feature count and iterations must
also match to align streams. K-Means restart seeds are derived from each algorithm seed.
Data seed is independent. Plot jitter uses seed 0 cosmetically, without altering results.

## Verified small reproduction commands

Use a new root; these engineering checks do not replace pilot results:

```bash
BA_SMOKE_ROOT=$(mktemp -d /tmp/ba-smoke.XXXXXX)
.venv/bin/ba-clustering experiment smoke --output "$BA_SMOKE_ROOT"
.venv/bin/ba-clustering experiment smoke --output "$BA_SMOKE_ROOT" --resume
.venv/bin/ba-clustering plot "$BA_SMOKE_ROOT/smoke"
```

Diagnostics: .venv/bin/python scripts/audit_frequency.py --output /tmp/ba-frequency-new.json
requires an unused output file and runs only seeds 0,1. Evidence in
verification/frequency_diagnostics.json was produced with the same script and a different
unused filename. Keep the output under diagnostic labeling; data-point initialization is
an intervention, not a baseline option.

Historical results under results/{smoke,baseline_pilot,ofat_pilot,interaction_pilot} were
read without alteration. All 948 record/spec/data/summary checks and ten selected pilot
metric reruns are recorded in verification/pilot_audit.json. Selected reruns cover seed
9000: BA reference and K-Means on all datasets, f_min=1.5 on all datasets, and one grid
cell. This does not establish full rerun reproducibility of all pilots. No discovery or
confirmation run was launched. Formal experiment commands are intentionally not part of
this engineering quick start; freeze the protocol before executing configured formal IDs.

## Reproduction limits

Exact results were checked in the same environment. BLAS/OpenMP, hardware, dependency
versions, ordering and floating point can affect low-order bits and timing across machines.
Runtime and timestamps are intentionally variable. No thread policy was enforced in
historical pilots; record threadpoolctl information and environment in future formal runs.

Resume validates spec/data, not code/environment, rewrites metadata and discards failed
attempt records. Append flush is not fsync; single-writer operation is assumed. Archive
outputs and source/dependencies before formal runs. A manifest alone is not a complete
immutable reproduction package. SHA256 evidence records the preserved historical files;
a final source snapshot is in verification/source_sha256.json.


## Notebook environment

Optional pyproject group notebooks declares ipykernel/nbformat/nbclient; installation
attempt failed DNS and no such package was installed. requirements-lock.txt remains the
verified base environment snapshot. scripts/run_notebook.py executes ordinary Python cells
without those packages and embeds actual PNG/table outputs. This verifies computations,
not Jupyter protocol or GUI behavior. See NOTEBOOK_GUIDE.md and notebook_audit.json.


Publication scope: the original workspace retains all historical raw pilot and notebook
session directories. The GitHub publication snapshot excludes those bulk generated
results and all private/user datasets; it includes the executed synthetic notebook and
selected audit evidence under docs/verification. A fresh clone will not contain the
historical raw pilot files. Reproduce engineering checks into fresh output directories;
do not present newly generated runs as the original historical evidence.
