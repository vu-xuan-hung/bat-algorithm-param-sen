# Handoff for a session with zero prior context

Last verification: **2026-11-05**, continuation after network access was restored.
Canonical Git repository: **/home/vxh/Desktop/New Folder/bat-algorithm-param-sen**.
Branch: **main**, origin: **https://github.com/vu-xuan-hung/bat-algorithm-param-sen.git**.
Verified published source snapshot: **19f20b29271769e7c26d9aa42e3c33c00db4d5d6**.
The follow-up commit containing this verification handoff changes only docs/evidence;
use `git rev-parse HEAD` and `git ls-remote origin refs/heads/main` for its exact current SHA.

The user selected this NEW destination after the prior bat-algorithm publication attempt.
It was verified empty before push, so no remote history was replaced. Snapshot 19f20b2
and the original bundle were verified; ls-remote and GitHub API confirmed the published SHA.
The older bat-algorithm repository received no writes in this continuation.

Original archive: **/home/vxh/Desktop/New Folder/ba-clustering**, without .git; its historical
research output directories and parent group documents stay untouched. Current docs/evidence
are mirrored there for continuity. Use the canonical clone for Git operations. Sections
below preserve the earlier audit history; the latest verification section supersedes its
network/fresh-install/kernel limitations. Do not mistake historical failure reports for
current blockers.

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

## Known bugs and limitations at the original audit

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

## Historical continuation prompt (superseded by latest section below)

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


## Latest verified continuation: GitHub, clean install and real kernel

Network now works in the primary session. GitHub authentication was validated without
exposing credentials. The resumed subagent retained its old network restrictions, so the
primary agent performed the actual clone/fetch/push. The subagent provided a read-only
protocol/code review; no additional agents or remote writes to the old destination.

- Original source directory: no Git metadata or unsaved work was overwritten. Bundle
  verify/list-heads matched 19f20b2; the offline snapshot is the root of the new repository.
- Clean Python **3.12.13** venv with **no system site packages**: installed setuptools/wheel,
  unchanged requirements-lock.txt, editable package, and .[notebooks]; **pip check passed**.
- Clean-env **372 tests passed in 4.98s**; separate **8 integration tests passed** with
  364 deselected; Ruff clean, **34 files formatted**, strict mypy clean on **16 modules**.
  CLI help/list/smoke also passed. Audit: verification/clean_install_verification.json;
  exact output: verification/clean_install_checks.txt. Environment freeze is evidence,
  not a replacement for the historical lock.
- Real **nbclient/ipykernel Jupyter execution** with the clean interpreter passed schema
  validation before/after, **10 code cells**, **48 successful records** (28 OFAT/20 grid),
  **11 inline PNGs**, **no errors**, seeds **0,1** only. Separate executed artifact:
  verification/parameter_sensitivity.jupyter.ipynb; audit: jupyter_kernel_audit.json.
  Outputs went to a unique /tmp directory. Original notebook outputs were not replaced.
  **VS Code GUI interaction remains unverified**; real kernel execution is now verified.
- No src/, tests/, configs/, BA semantics or original dependency lock changed. No new
  algorithm or formal seed execution. All original 948 research records remain preserved;
  compare hashes in verification/pilot_audit.json rather than re-running historical outputs.

### Remaining bounded tasks

1. Review docs/FORMAL_PROTOCOL_PROPOSAL.md with the researchers: normalization, 33-config
   pool, grids, paired inference, proposed practical thresholds, failure policy, power and
   actual stock-runner budget. It is **NOT FROZEN / NOT RUN / NO EXECUTION AUTHORIZATION**.
2. Before formal analysis, implement/test configuration deduplication for analysis (or a
   separately accepted execution registry), Q/Z endpoints, complete-seed checks and the
   declared Holm families. Today's CSV exporter does not implement this proposed protocol.
3. Add reviewed confirmation definitions with distinct original-reference/candidate roles;
   current confirmation template alone does not compare a tuned candidate with original C0.
4. Resolve resume provenance and schema validation gaps (historical T2/T4) before long
   runs, or establish a reviewed external immutable archive/identity gate.
5. Optional manual acceptance: open the source notebook in VS Code with the fresh kernel;
   verify GUI display. GUI acceptance is separate from the completed real-kernel execution.

T3 fresh installation/source publication is now verified for this environment; cross-
platform/Python-version testing and an offline wheel bundle remain outside that evidence.
No formal phase should run until its freeze/analysis/approval requirements are satisfied.

### Safe commands and next-agent prompt

In the canonical clone, inspect before writes:

```bash
git status --short --branch
git remote -v
git fetch origin
git log --oneline --graph --all -8
git ls-remote origin refs/heads/main
```

If DNS fails, local status/log and bundle verification remain safe. Retry fetch/push only
after connectivity returns; never use force, reset --hard, overwrite historical outputs,
or create a replacement root commit. Use the current GitHub status document for the
verified destination and snapshot. Newly authored verification docs can be pushed as an
ordinary fast-forward continuation of main; unrelated future commits need authorization.

Copyable prompt: Continue the existing project in the canonical bat-algorithm-param-sen
clone; read AGENTS, this latest handoff, ALGORITHM_SPEC and FORMAL_PROTOCOL_PROPOSAL.
Verify current HEAD/origin and actual evidence. Clean install, 372 tests/8 integration and
real Jupyter 10 cells/48 records/11 PNGs passed. Preserve original archive/results and BA
semantics. Start with protocol review/analysis prerequisites; do not run discovery or
confirmation, alter rosters, force-push or commit unrelated work without authorization.
