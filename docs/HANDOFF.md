# Handoff for a session with zero prior context

Last verification: **2026-11-05**. Last verified commit: **unavailable**; no usable Git
repository exists in the project or parent. No commit or Git initialization was performed.
Project root: /home/vxh/Desktop/New Folder/ba-clustering. Parent group documents are
source material and were preserved. Read ../ group files only when investigating their
claims; do not rewrite them as an incidental code task.

## Current state and completed work

The prior Claude implementation already contained BA, SSE, centroid encoding, K-Means,
OFAT/grid/repeated runs, resume, JSONL/CSV/statistics, plots, tests and saved pilots.
All 16 source modules, tests, configs and six existing Markdown files were audited.
README was zero bytes; AGENTS, CHANGELOG, INDEX, overview, API/data, experiment/results,
reproducibility, testing, troubleshooting and handoff documents were missing.
No Git history or tracked/untracked distinction could be obtained. Preserve all files
as existing work; before/after source hashes identify takeover changes.

Takeover fixed only K-Means configuration validation and experiment preflight validation.
It added a smoke-seed observer diagnostic, 29 tests and documentation/evidence.
It preserved velocity sign and every BA update/random-number convention. Historical
results and parent DOCX/XLSM/ZIP files remain unchanged. Original README write rejection
was respected: an exact draft was prepared and the user explicitly approved it before
writing. Five other rejected paths were absent, so new files replaced no earlier content.

## Verification actually executed

- Original full suite: 340 passed in 4.79s.
- New focused regressions before fixes: 23 failed,132 passed (confirming the defects).
- After fixes/additions: 369 passed in 3.85s; separate integration: 7 passed,362 deselected.
- Ruff: all checks passed; format: 32 Python files already formatted; mypy: no issues in
  16 source files. Final repeated command log is verification/checks.txt.
- Saved smoke=8, baseline_pilot=60, ofat_pilot=780, interaction_pilot=100 records, all
  completed; current config/data hashes and summaries match. Ten selected pilot reruns
  matched returned metrics except runtime exactly, including full BA histories. This is
  not a complete rerun of all pilots. Evidence: verification/pilot_audit.json.
- Twenty-four diagnostic runs used ONLY smoke seeds 0,1; no formal seed used. Reference
  plus zero-frequency/f_max=10/data-point-initialization interventions. Unchanged-production
  traces checked; evidence: verification/frequency_diagnostics.json.
- CLI help/list/run; smoke, resume, overwrite refusal, plots in a fresh temporary root;
  scripts' help; Python snippets and local documentation links checked. See checks.txt.

## Known bugs and limitations

The two confirmed validation defects above are fixed. No BA equation defect was found.
Tests do not prove complete correctness. Low-level optimizer helpers trust typed floating
arrays/states/draws; the public estimator performs clustering validation. Malformed nested
YAML/programmatically built ExperimentSpec and empty/duplicate seed lists are not fully
validated. Runtime does not enforce discovery/confirmation stage locks. Resume ignores
software changes and evaluation-label changes, rewrites original manifest environment,
and removes failed attempt records. Crash append flush is not fsync; no concurrent-writer
lock. Final centroids/labels are not persisted. High-dimensional grids plot only two factors.
A fresh network dependency installation is unverified; an offline clean editable install
failed because setuptools.build_meta was missing, and default uv cache writes were denied.
Existing .venv works. No dependencies were missing for the tests.

## Scientific concerns and decisions

Read ALGORITHM_SPEC, DECISIONS and SCIENTIFIC_AUDIT. Baseline velocity is
v+(position-best)*frequency, not the opposite sign. Local replaces global, strict own-bat
improvement plus loudness gate, accepted-only global best, sequential updates, mean-A
snapshot, clipped candidates/unclipped persistent velocities, fixed pre-drawn randomness.
Initial operational pulse equals r0; t starts at 1 and r schedule runs on acceptance.
Initial population is uniform inside processed-data bounds. Empty clusters are counted,
not repaired. SSE remains the sole objective. No K-Means refinement or alternate algorithm.

Historical OFAT: 230/1100140 global candidates accepted, 161694/1127360 local accepted.
Fresh reference diagnostics: 3/8783 global accepted; only three global proposals improve,
so the improvement test is the primary bottleneck in those runs. Frequency has a real
route to outcomes, but many paired trajectories tie. Effects depend on seed/data/settings
and initialization. No universal ineffectiveness or superiority over K-Means is established.
Literature comparisons inherited in older docs were not source-checked in this takeover;
verify the original papers before using those statements in a thesis.

## Important files and working commands

README and INDEX navigate all docs; ALGORITHM_SPEC is normative; config.py/bat_algorithm.py
implement the invariants; fitness.py is SSE; experiments.py/results.py implement runs and
I/O; plots.py reads saved results. configs/default.yaml and experiments.yaml contain
reference levels, datasets, budgets and stage seed rosters. Source hashes are in
verification/source_sha256.json; preserved historical file hashes in pilot_audit.json.

Run from project root:

```bash
.venv/bin/python -m pytest
.venv/bin/python -m pytest -m integration
.venv/bin/ruff check src tests scripts
.venv/bin/ruff format --check src tests scripts
.venv/bin/mypy
.venv/bin/ba-clustering --help
.venv/bin/ba-clustering list
.venv/bin/ba-clustering run --dataset iris --seed 0 --set max_iterations=5
```

Use fresh /tmp outputs for smoke/resume/plots as shown in REPRODUCIBILITY. Diagnostics
script takes --output pointing to a nonexistent file; it refuses overwrite. Scripts'
no-argument defaults launch pilots, so use --help first. Do not run formal experiments
as engineering checks. Failed commands: Git status/branch/log (no repository); isolated
offline pip editable install (missing backend); uv default cache (read-only lock directory).
Expected refusal: re-running an experiment into nonempty output without resume/overwrite
returns CLI code 2, preserving files. Matplotlib cache warning may appear; use writable
MPLCONFIGDIR if needed. No failed scientific run was concealed.

## Concrete next tasks (not authorized formal experiments)

| ID | Objective and files | Dependencies | Acceptance criteria | Required checks |
|---|---|---|---|---|
| T1 | Freeze scientific protocol and candidate-selection rule; docs/EXPERIMENT_DESIGN.md, configs/experiments.yaml | Discuss mechanism evidence with researchers; source-check literature claims | Written primary endpoint, effect threshold, multiplicity family, thread/cost policy; two specified 3x3 grids; separate original reference and candidate; confirmation seeds unseen | Config expansion/roster tests; smoke only; review freeze before formal runs |
| T2 | Preserve resume provenance and detect changed software/evaluation data; results.py, experiments.py, tests/test_experiments.py, REPRODUCIBILITY | Decide backward-compatible schema/migration; archive samples | Resume refuses mismatched software/data or explicitly records versioned attempts; original environment/start and failed attempts retained; old results readable | Interrupted/failure/resume/mismatch regressions on temp copies; full suite and static checks |
| T3 | Establish immutable source/dependency baseline; pyproject.toml, requirements-lock.txt, REPRODUCIBILITY, verification | Explicit user request before Git initialization/commit; package access for install | Fresh isolated install/import/CLI/tests pass; build backend recorded; source archive/hash or requested commit recorded; historical results unchanged | Fresh environment install with exact versions, all tests/lint/type; hashes |
| T4 | Strict spec/schema validation before I/O; config.py, experiments.py, datasets.py, tests/test_config.py, tests/test_experiments.py | Preserve valid current schema and error behavior | Invalid nested structures, empty/duplicate seeds, unsupported algorithms, invalid dataset metadata fail clearly before directory creation; runtime stage misuse gets an agreed guard | Invalid YAML/programmatic specs, untouched output, valid shipped configs, full suite |
| T5 | Save final solutions for independent SSE verification; experiments.py, results.py, DATA_CONTRACT, RESULTS_GUIDE | Agree additive schema and storage budget; retain legacy reading | New records include processed centroids/labels or artifact references/hash; independently recompute SSE without rerunning; legacy histories still load | Round-trip/shape/finiteness/SSE/legacy regression tests; smoke export and plots |

Do T1 and T3 before spending formal budgets. T2/T4 improve trustworthiness before long
runs. Do not solve the scientific concern by quietly changing the sign or initialization.
A separately requested variant would require its own module/config/tests/decision record.

## Acceptance criteria for a future continuation

Preserve all original outputs and group documents; never use formal seeds for debugging.
Show a failing regression before a defect fix, then relevant/full tests and static checks.
Record exact changed files, measured commands, evidence and unresolved limits. Update
AGENTS, CHANGELOG and this handoff. Do not claim confirmed sensitivity findings from pilots.
If prior file-write rejection would be bypassed by substantial replacement, obtain approval
of a concrete reviewed change. Do not commit without a user request.

## Ready-to-copy continuation prompt

Continue the existing ba-clustering project in /home/vxh/Desktop/New Folder/ba-clustering.
Read AGENTS.md, docs/HANDOFF.md, ALGORITHM_SPEC.md, DECISIONS.md and SCIENTIFIC_AUDIT.md.
Audit actual files; there is no usable Git history as of 2026-11-05. Preserve BA semantics,
all historical results and parent research documents. Original 340 tests passed; takeover
369 tests and 7 integration tests passed, with lint/format/mypy clean. Inspect recorded
evidence before claiming reproduction. Start at T1 protocol decisions and T3 installation
verification; do safe inspection and smoke checks only. Do not execute discovery 1000..1009
or confirmation 2000..2019, change the velocity sign, introduce a hybrid, initialize Git
or commit without explicit authorization. Resolve the next bounded task, run its regression
checks, and update the handoff with actual results and remaining limitations.


## Latest continuation: notebook and requested publication

The user additionally requested an ipynb comparing methods/parameter sensitivity and a
dataset folder accepting a path. Added notebooks/parameter_sensitivity.ipynb (10 executed
code cells, 11 embedded plots), datasets/example_blobs.csv (90x2 synthetic, metadata),
datasets/README.md, docs/NOTEBOOK_GUIDE.md, scripts/run_notebook.py and tests/test_notebook.py.
Default run: 28 OFAT+20 grid records, seeds 0,1 only. A custom labelled one-feature CSV,
K=2, is covered by integration execution. Notebook uses existing algorithms unchanged.
Latest suite is 372 passing including 8 integration; static checks remain clean. Fresh
Jupyter dependency install failed DNS; execution was sequential Python, not Jupyter kernel.
Output includes real PNG/table streams and runtime metadata, no simulated evidence.
Latest notebook execution evidence is recorded in verification/notebook_audit.json.

User explicitly requested **one subagent** after completion to push to
https://github.com/vu-xuan-hung/bat-algorithm. Necessary commit/push is now authorized for
that destination. Preserve remote history; no force-push. Include only project sources,
docs, tests, configs, synthetic example and selected evidence; exclude private user CSVs,
parent documents, virtualenv/caches and bulk generated results. Initial git ls-remote
failed: Could not resolve host github.com. Publication status must be verified, never
inferred from a local commit. See verification/publication_status.md when available.


Publication scope: the original workspace retains all historical raw pilot and notebook
session directories. The GitHub publication snapshot excludes those bulk generated
results and all private/user datasets; it includes the executed synthetic notebook and
selected audit evidence under docs/verification. A fresh clone will not contain the
historical raw pilot files. Reproduce engineering checks into fresh output directories;
do not present newly generated runs as the original historical evidence.
