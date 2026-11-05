# Implementation Plan

> Historical audit/plan below is preserved. Latest continuation verified a clean online
> installation and real Jupyter kernel, and published snapshot 19f20b2 to the new
> bat-algorithm-param-sen repository. See HANDOFF.md and verification/publication_status.md
> for current state; the earlier fresh-install/Git limitations below are superseded.


> Written **before** any implementation code, as required by the project brief.
> It records what the audit found and what was intended. The status section at
> the end was updated after implementation. For what the code does today, read
> [ALGORITHM_SPEC.md](ALGORITHM_SPEC.md) and [HANDOFF.md](HANDOFF.md).

## 1. Existing repository status (audit, 2026-11-05)

| Item | Finding |
|---|---|
| Working folder | `/home/vxh/Desktop/New Folder/` |
| Git | **Not a git repository.** `git status` fails with `fatal: not a git repository`, despite session metadata implying otherwise. No git repo was initialised, because that was not requested. |
| Python code, tests, dependency files, datasets, configs | **None.** Nothing to reuse and nothing to conflict with. |
| Project instructions | No `README`, `AGENTS.md` or `CLAUDE.md` existed. |

The folder held four project documents. They were read, not modified, and they
stay in the parent folder:

| File (in the parent folder) | Relevance |
|---|---|
| `BA_Clustering_Parameter_Sensitivity_Report_Opening.docx` | Group report draft: equations, centroid encoding, empty-cluster and boundary statements. |
| `Group_7_Literature_Review_BA_Parameter_Sensitivity.docx` | Proposed protocol: pilot budget, OFAT, seeds, statistics. |
| `Group_7_Eight_Week_BA_Parameter_Sensitivity_Roadmap.xlsm` | Tasks W1.1–W8.4 and an "Evaluation Protocol" sheet, including the reference configuration. |
| `Papers-20261105T091221Z-1-001.zip` | Five papers plus a Vietnamese synthesis. |

**Decision:** the project is created in a new subfolder, `ba-clustering/`, so the
group's documents stay untouched and the code path has no spaces.

### 1.1 Reusable inputs taken from those documents

These are design inputs, not results.

- **Reference configuration** (Protocol sheet): `A0=0.5, r0=0.5, alpha=0.9,
  gamma=0.9, f_min=0, f_max=2`. The sheet calls it "reference for screening only,
  not a proven optimum". It lives in `configs/default.yaml`, not in code.
- **Pilot budget:** 30 bats and at most 3,000 SSE evaluations per run.
- **Design:** OFAT with at least 3 levels per parameter, at least 10 shared
  discovery seeds, two 3×3 interaction grids, at least 20 unseen confirmation
  seeds, multi-start K-Means.
- **Logging requests:** boundary repairs (W4.1), boundary hits (W5.2), plateau
  evaluations, median/IQR/CI (W7.3).

### 1.2 Conflicts detected

The project brief takes precedence in every case. Conflicts are recorded rather
than silently resolved, and are carried into [DECISIONS.md](DECISIONS.md).

| # | Source says | Brief says | Resolution |
|---|---|---|---|
| C1 | Report §2.2.2: an empty cluster's centroid "is re-initialized" to a data point. | §4.3: no mutation during fitness; SSE stays defined; report the count; any repair is a separate later component. | Follow the brief. **The report text must be edited** to match. |
| C2 | Report §2.2.3: bats update "toward the current global best". | §3.2: `v + (X_i − X*) f`, sign preserved. With `f > 0` that term points **away** from `X*`. | Follow the brief and document the behaviour. **The report wording is inaccurate for this equation.** |
| C3 | Literature review and roadmap: `f_min < f_max`. | §5: `f_min <= f_max`. | Code allows equality. Shipped sweep levels are strictly ordered, which satisfies both. |
| C4 | Roadmap: `gamma >= 0`. | §5: `gamma > 0`. | Follow the brief. |
| C5 | Zhu et al. and Komarasamy & Wahi pseudocode: accept if `rand < A_i & f(x_i) < f(x*)` (global best). | §3.5: strictly better than **the bat's own** current fitness. | Follow the brief. Recorded as a literature ambiguity. |
| C6 | Carvalho et al. eq. 6 is recursive (`r^t = r^{t-1}[1−exp(−γt)]`), eq. 4 searches around `x_i`, and eq. 3 uses `v^{t−1}`. Fong et al. eq. 9 uses `x* + 0.01·rand()`. | §3.4, §3.7 | Follow the brief. Recorded. |
| C7 | Papers: `A0` typically in `[1, 2]`; Carvalho used 1–2. | §5: `0 <= A0 <= 1`. | Follow the brief. `A0 = 1` makes the loudness gate pass on every draw until the first decay. |
| C8 | Report: bat encoded as a `1×(K·D)` vector. | §4: shape `(K, D)`. | Follow the brief. The two are equivalent up to reshaping. |

## 2. Proposed architecture

Layered, with dependencies pointing downward only:

```text
cli.py ─ scripts/*.py                 thin entry points
experiments.py ─ results.py ─ plots   orchestration and I/O
clustering.py (BatClustering) ─ baseline.py (K-Means)
bat_algorithm.py                      problem-agnostic BA; no clustering, no I/O
fitness.py  boundaries.py  initialization.py  metrics.py
config.py  validation.py  datasets.py  preprocessing.py
```

`bat_algorithm.py` takes an objective callable and knows nothing about clusters.
Each equation is a small pure function. One iteration is a function of `(state,
t, pre-drawn randomness)`, so tests can inject exact random values and check
hand-calculated results.

### Components to create

All of them, since nothing exists. Beyond the suggested layout, three additions:
`validation.py` (shared array validators and exceptions), `plots.py` (figure
logic for `scripts/generate_plots.py`) and `tests/conftest.py`.

## 3. Mathematical implementation decisions

Full text and rationale are in [DECISIONS.md](DECISIONS.md).

1. **Velocity sign:** `v + (X_i − X*) f`, exactly as specified.
2. **Iteration index:** the loop counter `t` runs `1..T`, and that value enters
   the exponent. `t = 0` is reserved for initialisation, so the first acceptance
   gives `r0(1 − e^{−γ})`, not 0.
3. **Operational pulse rate:** `r_i` starts at `r0`. The schedule applies on
   acceptance only. Consequence, documented and tested: the first acceptance
   usually *lowers* `r_i` below `r0`.
4. **One candidate per bat per iteration:** the local candidate *replaces* the
   global one when triggered. Evaluations are therefore exactly `P(1 + T)`, and
   `P=30, T=99` gives exactly the 3,000-evaluation pilot budget.
5. **Velocity persistence:** velocity updates persist even when the candidate is
   rejected, and are computed from the *accepted* position.
6. **Global best:** updated only from accepted candidates, within the sweep, so
   later bats see the newest `X*`.
7. **`Ā^t`:** snapshot at the start of iteration `t`.
8. **Fixed-consumption randomness:** all draws for an iteration are made up front
   whatever branch runs. Parameter settings then share the same random stream for
   a given seed, which is what paired-seed comparison needs.
9. **Boundaries:** clip candidates, leave velocity unchanged, and count clipped
   coordinates.
10. **Initial population:** uniform inside the data's bounding box; zero velocity;
    `A_i = A0`.
11. **Empty clusters:** SSE defined; no repair; count reported.
12. **Stopping:** fixed budget by default. Optional patience compares against the
    best at the last *meaningful* improvement, so slow drift cannot reset it.
    The stop reason is `patience`, never "converged to optimum".

## 4. Expected tests

Every item in §10 of the brief, with injected-randomness tests for the update
rules, plus:

- the documented toy iteration from [ALGORITHM_SPEC.md](ALGORITHM_SPEC.md);
- a shared-random-stream test across parameter settings;
- preprocessing equivalence with scikit-learn's `StandardScaler`;
- result-store overwrite and resume protection;
- end-to-end experiment reproducibility.

No test needs network access; Iris and Wine ship inside scikit-learn.

## 5. Potential risks

| Risk | Mitigation |
|---|---|
| The specified sign repels bats from `X*`, so global moves may mostly hit bounds and be rejected, leaving local search to do the work. | Keep the sign. Log clipped coordinates and accepted-local counts per iteration so the effect is measurable. Report it honestly. |
| Local-search step `ε·Ā` is scale-dependent. | Default z-score preprocessing; document the dependence. |
| `A → 0` makes acceptance impossible, freezing the search. | Documented; visible through `n_accepted` in the history. |
| OFAT cannot reveal interactions. | Grid design supported; limitation documented. |
| Same iterations ≠ same cost for BA and K-Means. | Count both separately and keep the units distinct. |
| Readers treat pilot output as findings. | Results guide separates observations from conclusions. Pilot and smoke results are labelled as such. |
| Threaded BLAS/OpenMP changes low-order K-Means bits. | Documented in [REPRODUCIBILITY.md](REPRODUCIBILITY.md); determinism is claimed per environment only. |

## 6. Step-by-step order

1. Foundations: `validation`, `config`, `fitness`, `boundaries`, `initialization`, then their tests.
2. BA core: pure update functions, iteration, loop, then update-rule and state tests.
3. Clustering integration: `BatClustering`, `datasets`, `preprocessing`, `baseline`, `metrics`.
4. Experiment infrastructure: `experiments`, `results`, `plots`, `cli`, scripts, configs.
5. Documentation, written from verified behaviour.
6. Final verification: full tests, ruff, mypy, smoke and pilot experiments, re-run documented commands.

## 7. Status after implementation

All steps were completed. See [HANDOFF.md](HANDOFF.md) for measured test results
and outstanding tasks.


## 8. Independent continuation audit (2026-11-05)

The historical 'all steps completed' statement in section 7 was too broad: README was
empty and ten required docs plus AGENTS/CHANGELOG were missing at takeover. Source/tests/
configs and pilot outputs did exist. The original 340 tests passed. Targeted validation
fixes and diagnostics bring the executed suite to 369 passing tests with 7 integration;
missing documentation is now created and README populated after explicit user approval.
No formal experiment was run. Fresh installation remains unverified. See HANDOFF and
SCIENTIFIC_AUDIT for current status, evidence, limitations and concrete next tasks.
