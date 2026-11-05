# Testing

Run from ba-clustering/ using the existing .venv. On 2026-11-05:

| Command | Executed outcome |
|---|---|
| .venv/bin/python -m pytest | Original 340 passed; after changes 369 passed |
| .venv/bin/python -m pytest -m integration | 7 passed, 362 deselected |
| .venv/bin/ruff check src tests scripts | All checks passed |
| .venv/bin/ruff format --check src tests scripts | 32 files already formatted |
| .venv/bin/mypy | No issues in 16 source files |

Tests require no network datasets. Integration marker is declared in pyproject.toml;
the complete default suite includes integration tests. It treats RuntimeWarning as error.
Some experiment/CLI/plot tests are unmarked but still exercise multiple modules.

## Coverage by responsibility

- test_ba_updates: equations, strict trigger/gate, one-based t, snapshot loudness,
  immediate best update, documented toy iteration and rejection semantics.
- test_state_consistency: recomputed SSE, monotone best, every-iteration consistency,
  accepted-state ownership, evaluation budget, patience behavior.
- test_fitness: toy SSE 2,1,5 at (2,8),(1.5,8.5),(2.5,7.5), purity, empty clusters,
  brute-force agreement, overflow error. Velocity fixture produces (0.5,-0.5),
  proposed position (2.5,7.5), covered in test_ba_updates.
- test_boundaries: coordinate clipping/counts, constant features, velocity persistence.
- test_config and test_clustering: parameter/data shapes, nonfinite inputs, assignment,
  preprocessing, K-Means, public API, metrics and new K-Means validation regressions.
- test_reproducibility: repeated seeds/refit, global RNG independence and fixed consumption.
- test_experiments: designs/configs, every shipped experiment, seed roster separation,
  output/summary/statistics, overwrite refusal, interrupted resume, CLI and PNG plots;
  new preflight regressions ensure later invalid seeds/K-Means configs leave no output.
- test_integration: end-to-end fits, six controls reaching runs, export and deterministic
  histories. test_frequency_diagnostics: observer preserves production trajectories,
  candidate accounting and zero-frequency mechanism.

Twenty-three new regression cases failed before the fixes (23 failed,132 passed in the
focused run); four invalid tolerance cases already passed. Tests were not weakened.
Final additions total 29 cases including two diagnostic tests.

## Additional verification

Ten selected historical pilot reruns and all 948 record/manifest/summary checks are
recorded in verification/pilot_audit.json. Instrumented 24 smoke-seed runs are in
verification/frequency_diagnostics.json. No reserved formal seed was executed.
CLI smoke, resume, overwrite refusal, plot and API/README examples were checked in a new
temporary output root. Historical result hashes remained unchanged.

Limits: passing tests do not prove global optimality or a literature-wide definition of
BA, complete YAML schema validation, arbitrary injected-state validity, concurrent-writer
safety, crash durability after power loss or formal experimental acceptance. No fresh
network dependency installation was completed. See REPRODUCIBILITY and HANDOFF.


## Notebook extension

Latest full suite: 372 passed, including 8 integration tests; 34 Python files formatted.
test_notebook.py verifies exclusive output refusal before execution, saved traceback with
unexecuted stale outputs cleared, and full notebook execution on a user CSV with a text
label column, one feature and K=2 (48 records, 11 PNGs). Default 90x2 synthetic CSV also
ran all 10 code cells and produced 11 inline PNGs. See NOTEBOOK_GUIDE and notebook_audit.
An intermediate integration check exposed a missing output-root override in the exported
notebook; the source was corrected and re-executed, with no test relaxation.
