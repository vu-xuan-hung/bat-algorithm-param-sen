# Architecture Decision Records

Each record states what was decided and why. Add a new record rather than editing the
decision of an old one; mark the old one superseded and link to its replacement. Records
without an explicit date or status were accepted on 2026-11-05.

"The brief" means the project specification this repository was built from. Its
requirements are summarised in [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) and encoded in
the tests.

| ID | Topic |
|---|---|
| [ADR-001](#adr-001-velocity-sign-convention) | Velocity sign convention |
| [ADR-002](#adr-002-pulse-rate-initialisation-and-iteration-index) | Pulse-rate initialisation and iteration index |
| [ADR-003](#adr-003-candidate-generation-and-acceptance-semantics) | Candidate generation and acceptance |
| [ADR-004](#adr-004-empty-cluster-policy) | Empty-cluster policy |
| [ADR-005](#adr-005-boundary-policy) | Boundary policy |
| [ADR-006](#adr-006-random-number-strategy) | Random-number strategy |
| [ADR-007](#adr-007-initialisation-method) | Initialisation method |
| [ADR-008](#adr-008-sse-definition) | SSE definition |
| [ADR-009](#adr-009-local-search-perturbation-and-mean-loudness) | Local-search perturbation and mean loudness |
| [ADR-010](#adr-010-stopping-criteria) | Stopping criteria |
| [ADR-011](#adr-011-k-means-baseline-and-cost-accounting) | K-Means baseline and cost accounting |
| [ADR-012](#adr-012-results-storage-and-overwrite-protection) | Results storage and overwrite protection |
| [ADR-013](#adr-013-repository-location-and-version-control) | Repository location and version control |

---

## ADR-001: Velocity sign convention

**Context.** The brief specifies $v_i^{t+1} = v_i^t + (X_i^t - X^*) f_i$ and prohibits
silently reversing it. Zhu et al. and Carvalho et al. also print this form. With $f_i > 0$,
the added term points away from $X^*$. The group report's prose (§2.2.3) says bats move
"toward" the best, and Fong et al. use similar wording, which does not match the equation.

**Decision.** Implement the equation exactly. Do not add a configuration option for the
sign.

**Alternatives considered.**
- *Use $(X^* - X_i)$, which moves toward the best.* Rejected for the baseline, as the brief
  requires. It would be a different algorithm.
- *Add a `sign` option.* Rejected: the core must stay standard, and an option invites an
  accidental change of baseline.

**Reason.** The baseline must match its specification, so that its results can be attributed
to it.

**Consequences.** Global moves repel from the best and are rarely accepted, and this
materially limits the frequency parameters' influence
([pilot observations](RESULTS_GUIDE.md#7-pilot-observations-not-conclusions)). **This is the
most consequential open research question in the project.** See task T1 in
[HANDOFF.md](HANDOFF.md).

**Related.** `velocity_update` in `src/ba_clustering/bat_algorithm.py`;
`test_velocity_sign_convention_from_the_brief` and
`test_the_global_move_points_away_from_best` in `tests/test_ba_updates.py`.

---

## ADR-002: Pulse-rate initialisation and iteration index

**Context.** The schedule $r_i = r^0(1 - e^{-\gamma t})$ equals zero at $t = 0$, yet $r^0$
is commonly called the initial pulse rate. Silently using it at $t = 0$ would make the
first accepted bat always use local search; silently replacing it would change the
algorithm.

**Decision.**
1. `r0` is the **reference parameter** of the schedule.
2. The **initial operational pulse rate** is defined separately and equals `r0`
   (`initial_pulse_rates`).
3. The loop counter `t` runs from 1 to $T$, and that value enters the exponent. $t = 0$
   labels initialisation and is never passed to the schedule during a run.
4. The schedule is applied **only on acceptance**, and it replaces the current rate.

**Alternatives considered.**
- *Initial rate 0, from the formula at $t = 0$.* This gives a monotone rate and has some
  support in the papers ("can be around zero"). Rejected because `r0` would then have no
  effect until a bat's first acceptance, and the group's report and parameter list call it
  the initial pulse rate.
- *Zero-based counter.* Rejected: the first acceptance would always set $r_i = 0$.
- *Apply the schedule every iteration.* Rejected: the brief requires an
  acceptance-dependent update.
- *Modified schedule that starts from `r0`.* Rejected: the brief prohibits replacing the
  equation.

**Reason.** It keeps the specified equation unmodified, and gives `r0` its described meaning
from the first iteration.

**Consequences.** The first acceptance usually **lowers** the rate below `r0`
(to $r0(1 - e^{-0.9}) \approx 0.593\,r0$ at $t = 1$ with $\gamma = 0.9$), after which later
acceptances raise it. So "pulse rate increases over time" is true only after the first
acceptance. After a first acceptance late in a run, the dip is small because $e^{-\gamma t}$
is tiny. Rates always stay within $[0, r0]$.

**Related.** `initial_pulse_rates` in `src/ba_clustering/initialization.py`;
`pulse_rate_update`; `test_documented_initial_pulse_convention_is_r0`,
`test_first_acceptance_can_lower_the_pulse_rate_below_r0`,
`test_full_run_passes_one_based_iteration_counter_to_the_pulse_schedule`.

---

## ADR-003: Candidate generation and acceptance semantics

**Context.** The brief fixes the two acceptance conditions, but leaves some sequencing open.
Published pseudocode leaves it open too, and two papers in the archive compare against the
global best instead.

**Decision.**
1. Accept only if the candidate's fitness is **strictly better than the bat's own current
   fitness** and a uniform draw is **less than the bat's current loudness**.
2. Each bat evaluates exactly **one candidate** per iteration. A local-search candidate
   replaces the global one.
3. Velocity is computed from the **accepted** position, and **persists even if the candidate
   is rejected**.
4. The global best updates **only from accepted candidates**, immediately, so later bats in
   the sweep see it.
5. The initial population counts as accepted.

**Alternatives considered.**
- *Compare with the global best in condition 1*, as in the Komarasamy & Wahi and Zhu et al.
  pseudocode. Rejected by the brief. It would also make acceptance far rarer.
- *Update the best from any evaluated candidate.* Rejected by the brief. A very good
  candidate failing the loudness draw is therefore lost.
- *Evaluate both a global and a local candidate.* Rejected: it makes cost depend on the
  random draws, and so on `r0`, which would break the equal-budget comparison.
- *Roll back velocity on rejection.* Rejected: it is not in the equations, and would freeze
  bats that are never accepted.
- *Synchronous sweep with one $X^*$ per iteration.* Rejected: the brief defines $X^*$ as the
  best discovered so far.

**Reason.** It follows the brief literally, and makes cost independent of parameter values.

**Consequences.** Cost is exactly $P(1 + T)$ evaluations. Velocity can grow without bound on
a bat that keeps being rejected (it reached about 60 in pilot runs on data spanning a few
standard deviations). This is finite and detected if it ever overflows.

**Related.** `run_iteration`, `should_accept`; `tests/test_state_consistency.py`;
`test_improving_candidate_rejected_by_loudness_changes_nothing_but_velocity`,
`test_later_bats_in_the_sweep_see_the_updated_best`.

---

## ADR-004: Empty-cluster policy

**Context.** A centroid may own no samples. The brief calls for SSE to stay defined with no
mutation during evaluation. **The group report (§2.2.2) instead says that an empty cluster's
centroid "is re-initialized" to a data point.**

**Decision.** Follow the brief. Assign by nearest centroid; compute SSE normally; never move
a centroid during evaluation; report the count as `n_empty_clusters`. There is no repair.

**Alternatives considered.**
- *Re-initialise, as in the report.* Rejected: it makes fitness evaluation impure, adds a
  mechanism outside standard BA, and consumes randomness. It may be added later as a
  separate, configurable component.
- *Penalise empty clusters.* Rejected: SSE must be the only primary fitness.

**Reason.** Purity makes evaluation reproducible and testable.

**Consequences.** BA can return fewer than $K$ effective clusters. In the pilot, 21 of 250
Wine BA runs kept two empty clusters out of three, which makes Silhouette undefined.
**The report text must be corrected to match, or a repair component must be added
deliberately.** See task T2 in [HANDOFF.md](HANDOFF.md).

**Related.** `compute_sse`, `count_empty_clusters`;
`test_sse_is_defined_with_an_empty_cluster`, `test_empty_clusters_are_reported_and_not_repaired`.

---

## ADR-005: Boundary policy

**Context.** The brief requires clipping and asks for explicit documentation of the details.

**Decision.** Clip every candidate, global or local, coordinate-wise to the preprocessed
data's per-feature range, before evaluation. **Do not change velocity.** Count clipped
coordinates. Keep NaN unchanged in clipping, so it is detected by the non-finite fitness
check.

**Alternatives considered.**
- *Zero or reflect velocity at the boundary.* Rejected: it is an unannounced correction
  mechanism, and would change the dynamics being studied.
- *Clip velocities.* Rejected: same reason.
- *Reject out-of-bounds candidates.* Rejected: most global moves would be wasted outright.

**Reason.** It is the simplest policy, and it matches the report's statement that
coordinates are clamped.

**Consequences.** Global candidates often land on the box boundary. Zero-range features are
pinned to their constant.

**Related.** `src/ba_clustering/boundaries.py`; `tests/test_boundaries.py`, especially
`test_clipping_does_not_change_velocity`.

---

## ADR-006: Random-number strategy

**Context.** Parameter sensitivity relies on paired seeds. If the amount of randomness used
depended on which branches ran, then changing `r0` would shift every later random number,
and pairing would only cover the initial population.

**Decision.** Use one `numpy.random.default_rng(seed)` per run, created in `fit`. Draw the
initial positions first. Then, each iteration, draw `beta`, `pulse_draw`, `epsilon` and
`loudness_draw`, in that order, whether or not they are used. Never use NumPy's global
random state.

Other generators exist only outside the optimiser, each with a clear reason:
`make_blobs` for synthetic data (seeded separately, because data must not vary across
algorithm seeds), and scikit-learn K-Means for its own restarts (restart seeds derived from
the run seed).

**Alternatives considered.**
- *Draw lazily, as needed.* Rejected: the stream would diverge across parameter settings.
- *Separate streams per bat.* Rejected: unnecessary complexity.

**Reason.** Common random numbers reduce comparison noise, and make some effects exact: a
parameter with no route to the outcome produces identical results.

**Consequences.** Slightly more draws are made than strictly needed; the cost is negligible.
**Reordering draws changes every result** even though the distribution is unchanged. Treat
it as a breaking change: record an ADR and rerun any results you rely on.

**Related.** `draw_iteration_randomness`; `tests/test_reproducibility.py`, especially
`test_draw_order_is_beta_pulse_epsilon_loudness` and
`test_run_consumes_the_same_stream_whatever_the_parameters`.

---

## ADR-007: Initialisation method

**Context.** The brief requires reproducible initialisation that respects feature bounds,
and leaves the method open.

**Decision.** Draw each coordinate uniformly within the data's bounding box. Start velocities
at zero, and every bat at $A_i = A0$, $r_i = r0$.

**Alternatives considered.**
- *Sample data points as centroids (Forgy).* Rejected as the default: it uses information
  about the data's distribution that BA's standard random initialisation does not, and
  would start much closer to good solutions.
- *K-Means++ seeding.* Rejected: a form of hybridisation, which is out of scope.
- *Randomise $A_i$ and $r_i$ per bat*, as in Carvalho et al. Rejected: it would blur the
  meaning of the studied parameters.

**Reason.** It is the conventional box initialisation, and it adds no data-driven advantage
that could confound the comparison.

**Consequences.** Early best fitness is poor (mean of about 425 on z-scored Iris), and
initial centroids may own no samples. The K-Means baseline does get a data-driven start
(k-means++), which is a real asymmetry to report.

**Related.** `initialize_positions`; `test_initial_positions_have_the_required_shape_and_respect_bounds`,
`test_initial_positions_come_first_from_the_stream`.

---

## ADR-008: SSE definition

**Decision.** $SSE = \sum_n \min_k \lVert x_n - \mu_k \rVert_2^2$, total and squared. It is the
only optimisation objective. Silhouette and ARI are evaluation-only.

**Alternatives considered.** Unsquared distances, mean instead of total, negative SSE,
accuracy, and Silhouette as a fitness: all rejected by the brief.

**Reason.** It is the K-Means objective, so the two algorithms optimise the same quantity.
Using `min` makes it well defined with empty clusters.

**Consequences.** Lower SSE need not mean better clustering, and SSE values are only
comparable across runs with the same data and preprocessing. K-Means's SSE is recomputed by
this project's function rather than taken from `inertia_`, so both algorithms share one
implementation.

**Related.** `src/ba_clustering/fitness.py`; `tests/test_fitness.py`.

---

## ADR-009: Local-search perturbation and mean loudness

**Decision.** $\epsilon$ is a `(K, D)` array of independent draws from $U[-1, 1)$, with no
further scale factor. $\overline{A}^{\,t}$ is the mean of all $A_i$, snapshotted at the start
of iteration $t$.

**Alternatives considered.**
- *A single scalar $\epsilon$ per candidate.* Rejected by the brief.
- *An additional step factor, like Fong et al.'s 0.01.* Rejected: it is not in the specified
  equation.
- *Scaling by feature range.* Rejected: same reason.
- *A live mean, updated as bats accept.* Rejected: the superscript $t$ denotes a per-iteration
  value, and a snapshot makes results independent of the order bats accept in.

**Consequences.** The step size is in data units, so preprocessing changes the search
behaviour. Under the reference settings, the mean loudness stays around 0.2–0.25 at the end
of a run on z-scored data, meaning steps of up to about ±0.2 per coordinate. This may limit
fine convergence, and it should be considered when reading the gap to K-Means.

**Related.** `local_search_candidate`; `test_mean_loudness_is_a_snapshot_from_the_start_of_the_iteration`.

---

## ADR-010: Stopping criteria

**Decision.** The default is a fixed iteration budget. Optional `patience` stops after that
many consecutive stalled iterations, where stalled means best fitness has not improved by
more than `tolerance` since the last meaningful improvement. Setting `tolerance > 0` without
`patience` is rejected. The reported reason is `patience`, never "converged".

**Alternatives considered.**
- *Compare consecutive iterations only.* Rejected: tiny gains would reset the counter forever.
- *Ignore a useless tolerance silently.* Rejected: the brief prohibits silent fixes.
- *Stop on evaluations.* Unnecessary: evaluations follow directly from iterations.

**Consequences.** Early-stopping runs use a variable budget, so formal comparisons should
keep the fixed budget.

**Related.** `run_bat_algorithm`; `test_patience_stops_after_consecutive_stalled_iterations`,
`test_slow_drift_does_not_reset_patience`.

---

## ADR-011: K-Means baseline and cost accounting

**Context.** scikit-learn's `n_init` reports iterations for the best restart only, and
iteration counts of the two algorithms are not comparable.

**Decision.** Run restarts individually (`n_init=1` in a loop), with restart seeds derived
from the run seed. Use `algorithm="lloyd"`, and keep the lowest-SSE restart. Report
`n_lloyd_iterations` across all restarts, and keep it in a separate column from BA's
`n_evaluations`. The baseline defaults to 10 restarts (multi-start, per the group's protocol),
and each restart's SSE is recorded.

**Alternatives considered.** A single K-Means run: rejected as the only baseline, since the
protocol specifies multi-start, but `restart_sse` still exposes single-restart outcomes.
Reporting equal "iterations": rejected by the brief.

**Consequences.** One Lloyd iteration and one SSE evaluation each make roughly one full
assignment pass, but the Lloyd iteration also recomputes centroids, and k-means++ seeding is
not counted. The columns are close in order of magnitude only, not equivalent.

**Related.** `src/ba_clustering/baseline.py`; `test_ba_cost_is_not_reported_as_kmeans_cost`.

---

## ADR-012: Results storage and overwrite protection

**Decision.** Each experiment writes `results/<id>/`. Runs are appended to `runs.jsonl` as they
finish, and the CSV tables are written at the end. Existing results are never replaced
without `--overwrite` or `--resume`. `--overwrite` refuses any directory lacking a manifest.
`--resume` refuses a changed definition, by comparing a hash of the definition and data.
Figures are written to a `figures/` subdirectory.

**Alternatives considered.** A database: unnecessary. One JSON file written at the end:
rejected, as a crash would lose everything.

**Consequences.** Results survive interruption. Resuming under a changed software environment
is **not** detected: the manifest records only the latest environment.

**Related.** `src/ba_clustering/results.py`; `test_results_are_never_overwritten_without_an_option`,
`test_overwrite_refuses_a_directory_that_is_not_results`,
`test_resume_runs_only_what_is_missing_and_matches_a_full_run`.

---

## ADR-013: Repository location and version control

**Context.** The working folder held the group's documents, and was not a git repository.

**Decision.** Create the project in a `ba-clustering/` subfolder. Leave the group's documents
unmodified in the parent folder. Do not run `git init`, because it was not requested.

**Consequences.** `git_commit` is `null` in all current results, so `software_version` is the
only code identifier. Initialise git before formal experiments: see task T3 in
[HANDOFF.md](HANDOFF.md).


---

## ADR-014: Validate every planned seed and baseline configuration before result I/O

**Date/status:** 2026-11-05, accepted; implementation defect fix, no algorithm change.

**Problem:** preflight checked BA only with the first seed. A later negative seed or
invalid K-Means configuration could create a results directory before raising. KMeansConfig
also accepted boolean/fractional/nonfinite integer settings and infinite tolerance.

**Decision:** validate BA configurations for every seed and K-Means settings for every
seed/dataset before prepare_directory. Require non-boolean integral K/seed/restarts/budget
and finite float tolerance at K-Means configuration construction. Preserve valid defaults
and the separate cost units. Regression tests reproduce the prior failures.

**Limits:** this does not introduce a complete YAML schema or validate arbitrary manually
constructed ExperimentSpec objects. T4 in HANDOFF describes that remaining work.

---

## ADR-015: Diagnose frequency behavior without changing the baseline

**Date/status:** 2026-11-05, accepted.

**Decision:** scripts/audit_frequency.py observes the objective calls made by production
run_iteration, counts improving/gate-rejected/accepted branches, and compares standard
initialization best-fitness traces and final best positions with the ordinary estimator.
Draw no extra production randomness. Use only smoke seeds; label initialization/frequency
interventions separately and exclusively create diagnostic output files.

**Reason:** distinguish the improvement test from the loudness gate and algorithm behavior
from defects without substituting another BA variant. Historical outputs remain unchanged.
No velocity reversal, refinement or alternative implementation was introduced.

**Evidence:** SCIENTIFIC_AUDIT.md and verification/frequency_diagnostics.json.

---

## ADR-016: Register larger UCI benchmarks as opt-in local CSVs

**Date/status:** 2026-11-11, accepted.

**Decision.** Use UCI's headered CSVs with the existing CSV loader. Pin the original
bytes by SHA-256 in the download script and store the files in the ignored `datasets/`
directory. Set K from the published class count and use class labels for ARI only.
Use the base configuration's z-score preprocessing. Preserve the full rows, feature
order and natural class frequencies. Include new datasets in formal designs only after
the protocol is reviewed and frozen.

**Reason.** These benchmarks increase data size, feature count or K and cover class
imbalance while using the same preprocessing, optimizer and evaluation pipeline.
