# Instructions for research agents

Mission: continue **Application of Bat Algorithm to Data Clustering and Parameter
Sensitivity Analysis**. Establish a trustworthy specified Standard Continuous BA baseline
and study A0, r0, alpha, gamma, f_min, f_max. Read docs/HANDOFF.md, ALGORITHM_SPEC.md,
DECISIONS.md and TESTING.md before editing. Inspect actual source and current results.

## Structure and state

Python package: src/ba_clustering; tests: tests; entry points: cli.py and scripts;
configurations: configs; documentation: docs; permanent observations: results.
Group source documents live in the parent directory and must remain untouched.
BA, SSE, K-Means, OFAT/grid/repeated experiments, JSONL/CSV, resume and plots exist.
On 2026-11-05 the takeover verified 372 tests, 8 integration tests, lint, format and mypy.
The original ba-clustering/ archive has no .git. The canonical Git clone is the sibling
bat-algorithm-param-sen/ repository, on main, origin
https://github.com/vu-xuan-hung/bat-algorithm-param-sen.git. Snapshot 19f20b2 was verified
on GitHub after the user selected that new, empty destination. Preserve its history.
The user authorized the necessary verification/documentation commit and push in this
continuation; unrelated future commits still need user authorization. Never force-push.
Use verification evidence to distinguish historical snapshots from current state.

## Mathematical invariants

Preserve frequency f_min+(f_max-f_min)*beta, velocity v+(position-best)*f, proposal
position+updated_velocity. Local search replaces the global proposal when draw>r_i;
epsilon is independent U[-1,1) per coordinate; use mean loudness at sweep start.
Clip candidates, never velocities. Accept only strict improvement over the bat's own
fitness AND draw<A_i. Rejected candidates keep accepted position/fitness/A/r unchanged;
velocity persists. Update A and r only on acceptance; r starts at r0 and uses the
reference r0*(1-exp(-gamma*t)), t=1..T. Update best only from accepted candidates,
immediately within the sequential sweep. Best SSE never increases.
SSE is total squared distance to nearest centroid; empty clusters are reported without
repair. No K-Means refinement inside BA. Keep shapes (N,D), (K,D), (P,K,D).

Do not silently reverse the sign, rescale epsilon, align centroid permutations, alter
acceptance, draw order, initialization, clipping or pulse convention. No six convergence
factors, WOA, SCA, binary BA, feature selection or hybrid refinement. A separately
requested variant needs a separate implementation, config, tests and decision record.

## Development and verification

Use Python >=3.11 syntax, typed small functions and NumPy float64; current verification
uses 3.12.13. Reference research settings live in YAML. Keep edits narrow, preserve public
interfaces, and add a regression test that fails before fixing a confirmed defect.
Run .venv/bin/python -m pytest, .venv/bin/python -m pytest -m integration,
.venv/bin/ruff check src tests scripts, .venv/bin/ruff format --check src tests scripts,
and .venv/bin/mypy from this project root. Never weaken tests to pass.
Update the relevant spec/API/data docs with behavior changes; append decisions and
CHANGELOG, and refresh HANDOFF with actual commands, results and remaining gaps.
Do not call inspected tests executed tests. Report failed commands and numerical limits.

## Reproducibility and preservation

Use default_rng(config.seed), preserve fixed draw consumption and seed independence.
Smoke seeds 0,1 are for debugging; pilot seeds 9000..9009 are exploratory. Never run
reserved discovery 1000..1009 or confirmation 2000..2019 during debugging. Synthetic
data seed 42 is separate. Do not run formal experiments without protocol freeze.
Read historical results; never delete or replace them. Use a fresh /tmp output root for
checks. Resume rewrites JSONL and manifest, so never use it on historical evidence merely
for testing. Record software, preprocessing, dataset hash and config; current resume does
not enforce matching software environments. Do not claim superiority over K-Means.

README changes were explicitly approved during takeover. Five previously rejected docs
were absent and were created without replacing content. Any future substantial replacement
of existing documentation must be reviewed with the user when prior rejection applies.
Current limits and concrete next tasks are in docs/HANDOFF.md.


## Notebook continuation

User requested a notebook, path-based user CSV loading, and one subagent to push the
finished project; the destination was subsequently changed by the user to
https://github.com/vu-xuan-hung/bat-algorithm-param-sen. This authorizes the
necessary Git commit/push workflow for that destination; never force-push or replace
remote work. Inspect the remote first. Parent documents/user datasets/caches stay local.
Notebook: notebooks/parameter_sensitivity.ipynb; CSV examples: datasets/; runner:
scripts/run_notebook.py. Follow docs/NOTEBOOK_GUIDE.md. Original notebook outputs are real Python
cell execution. A separate executed artifact now verifies a real Jupyter kernel in a
fresh environment; VS Code GUI interaction remains unverified. Default seeds are 0,1;
results go to unique session directories. Notebook tests use temporary output roots.


Formal discovery/confirmation is still NOT authorized to run. Read
docs/FORMAL_PROTOCOL_PROPOSAL.md as a proposal, not a frozen protocol. Both new kernel
verification and clean-environment tests used only smoke seeds and temporary outputs.
No src/, tests/, configs/ or baseline lock content changed in this verification continuation.
