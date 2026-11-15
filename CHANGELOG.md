# Changelog

## 2026-11-11 — UCI benchmark datasets

- Registered Rice, Dry Bean, HTRU2 and Letter Recognition as local CSV datasets.
- Added a SHA-256-verified UCI downloader and documented CLI and notebook settings.

## 2026-11-05 — independently verified continuation

- Audited the existing implementation; original 340 tests passed. No usable Git history.
- Preserved all historical results and group documents. Did not run formal seed rosters.
- Fixed KMeansConfig integer/type and finite-tolerance validation.
- Fixed experiment preflight to validate every seed and K-Means configuration before I/O.
- Added 29 regression/diagnostic tests; final suite: 369 passing, including 7 integration.
- Added scripts/audit_frequency.py: observer around production iterations, exact trajectory
  checks, exclusive output creation, smoke seeds only. No BA semantic change.
- Validated 948 saved records and summaries; ten pilot reruns matched non-runtime metric
  fields exactly. Added docs/verification/pilot_audit.json and frequency_diagnostics.json.
- Created missing agent, testing, data/API, experiment, reproducibility, results and
  handoff documentation. Completed the empty README after explicit user approval.
- Appended clarifications to existing algorithm/decision/implementation documents;
  retained the architecture and mathematical notation.

Existing package version remains 0.1.0; these validation/documentation changes do not
identify historical code by a commit. No Git commit was created.


## 2026-11-05 — requested notebook and CSV workflow

- Added notebooks/parameter_sensitivity.ipynb with executed tables and 11 genuine plots:
  six-control OFAT, A0/r0 grid, BA/K-Means variation, convergence, acceptance and clusters.
- Added datasets/ with a 90-row synthetic CSV/provenance and user path/K/label instructions;
  user datasets are ignored by default. No algorithm variant or objective change.
- Added scripts/run_notebook.py for sequential Python execution/rich-output capture,
  output refusal and error preservation; 3 tests including custom one-feature labelled CSV.
- Declared optional notebook dependencies; installation attempt failed due DNS, so actual
  VS Code/Jupyter execution remains unverified. Base dependency lock unchanged.
- Latest checks: 372 tests, 8 integration tests, Ruff/format and mypy pass.
- User requested one subagent to push the finished project to vu-xuan-hung/bat-algorithm;
  remote connectivity/publication must be reported from actual evidence.


## 2026-11-05 — publication recovery and clean-environment verification

- User changed GitHub destination to vu-xuan-hung/bat-algorithm-param-sen. Verified it was
  empty, pushed unchanged snapshot 19f20b2 to main, and confirmed the SHA via GitHub API
  and ls-remote. Old destination and historical source archive were not modified by Git.
- Verified the README installation sequence in a fresh Python 3.12.13 venv using the
  original pinned runtime/dev lock, then notebook extras; pip check and all 372 tests,
  8 integration tests, Ruff/format/mypy passed.
- Validated/executed the notebook using an actual Jupyter kernel: 10 code cells,
  48 successful records, 11 PNGs. Stored separate executed artifact and audits, preserving
  prior notebook outputs and research results. VS Code GUI was not exercised.
- Updated current documentation and prepared a formal protocol proposal, not frozen or
  executed. No BA/source/test/config/legacy-lock change or formal seed consumption.
