# Proposed formal discovery and confirmation protocol

**DRAFT — NOT FROZEN — NOT RUN — NO EXECUTION AUTHORIZATION.**

Prepared from the source/configuration in the canonical clone
`/home/vxh/Desktop/New Folder/bat-algorithm-param-sen` on 2026-11-05. This is a proposal
for researcher review, not a statement that thresholds, hypotheses or budgets were
accepted. No formal algorithm seed was executed to prepare it. Reading configurations and
expanding/validating their conditions used no objective evaluations. Existing pilot data
remain exploratory and are not independent confirmation.

## 1. Objective, hypotheses and scope

Study the six controls of the existing specified Standard Continuous BA: A0, r0, alpha,
gamma, f_min and f_max. SSE remains the optimizer's sole objective; the sign
`v + (position - best) * frequency`, accepted-only A/r updates, operational initial r0,
local-candidate replacement, boundary clipping, population initialization and fixed random
consumption remain unchanged. No hybrid refinement, centroid permutation alignment or new
BA variant is introduced by this protocol.

Primary questions: which tested configurations improve normalized final SSE under a fixed
budget, and does one prospectively selected configuration retain a practically meaningful
advantage over the original reference on unseen algorithm seeds? OFAT estimates conditional
main effects around that reference. Two prespecified grids inspect particular interactions;
they are not a full six-dimensional search. Frequency effects and global/local acceptance
are mechanism outcomes, not a reason to alter the baseline to obtain a desired finding.

## 2. Fixed data, controls and costs

Use the three shipped datasets, frozen before discovery: Iris (150x4,K=3), Wine
(178x13,K=3), and synthetic blobs (300x4,K=4,std=1.5,data seed42). Data seed42 is distinct
from algorithm seeds. Fit z-score preprocessing once per dataset (population SD, ddof0),
share the processed matrix across algorithms, retain constant-feature handling, and store
processed data hashes, shape, feature order and preprocessing parameters. External class
labels are evaluation-only; preserve/hash them separately because the current matrix hash
does not cover labels. Dataset additions/replacements require a new protocol freeze before
any formal outcome is inspected. Confirmation on the same data but unseen optimizer seeds
tests stochastic robustness, not generalization to unseen datasets.

BA: P=30, T=99, patience=null, tolerance=0, exactly P(1+T)=3000 full SSE evaluations/run,
including 30 initialization evaluations. Original reference C0:
`A0=.5,r0=.5,alpha=.9,gamma=.9,f_min=0,f_max=2`. Every condition gets the same P/T and K.

K-Means: Lloyd, k-means++, ten explicit independent restarts derived from the run seed,
max_iter300, tol0.0001, lowest project-recomputed SSE retained. Report total Lloyd
iterations across all restarts and restart SSE. K-Means is a conventional quality baseline,
not an equal-SSE-evaluation-budget competitor; never equate its Lloyd iterations to BA
objective evaluations. Report runtime descriptively, with hardware/threads controlled.

Run serially on one frozen host/environment. Set OMP_NUM_THREADS, OPENBLAS_NUM_THREADS,
MKL_NUM_THREADS, BLIS_NUM_THREADS, NUMEXPR_NUM_THREADS and VECLIB_MAXIMUM_THREADS to1 before
Python/NumPy import. Record actual threadpoolctl libraries/versions/thread counts, CPU,
OS, Python and pinned dependencies. Set/report PYTHONHASHSEED0 as process metadata if used;
BA still uses only its local default_rng. Freeze process/run order and any warm-up policy;
do not compare timings from arbitrary concurrent notebook activity. No thread-setting or
warm-up command was executed as part of preparing this proposal.

## 3. Endpoint and aggregation: do not pool raw SSE

For each fixed processed dataset d, compute once
`TSS_d = sum_n ||x_nd - mean(X_d)||^2`. Reject a zero/nonfinite TSS dataset before freeze;
do not introduce an outcome-dependent epsilon denominator. Let
`Q(c,d,s) = final_SSE(c,d,s) / TSS_d`. Raw SSE remains the optimizer output and is reported
within dataset. Q is an analysis endpoint, not a change to fitness. It is dimensionless;
mean SSE/(N*D) would be similar for fully nonconstant z-scored data, but use the actual
frozen TSS so constant columns are handled explicitly.

For ranking one common configuration across these datasets:
`Score(c) = mean_d(mean_s Q(c,d,s))`. Give each of the three datasets equal weight,
regardless of N, D or raw SSE magnitude. Do not concatenate raw dataset SSE values and do
not treat 30 dataset-seed observations as 30 independent datasets. Keep per-dataset
results visible. The corresponding seed-block endpoint for paired aggregate inference is
`Z(c,s) = mean_d Q(c,d,s)`; the independent repetitions are algorithm seeds, conditional
on these three fixed datasets. Identical seed numbers across datasets are retained as a
block, rather than assuming within-block independence.

Secondary outcomes: within-dataset raw SSE, Silhouette (including undefined counts),
external ARI, empty-cluster count, complete best-SSE trace, evaluations-to-plateau,
wall-clock fitting time, local/global evaluated/accepted counts, clipping coordinates,
and max velocity. A plateau is budget-relative, not proof of optimality. Record failures
and runtime variability; do not optimize candidate selection on external labels.

## 4. Discovery seed roster and OFAT

Use exactly discovery algorithm seeds1000..1009 (ten paired repetitions), shared across
all configurations and algorithms. Do not inspect/use confirmation2000..2019 during
debugging, discovery, candidate selection or method development. Debug only with smoke0,1
or the already explored pilot roster9000..9009. Save all initial/iteration RNG semantics;
changing P, T, data dimensions or draw order would break the intended pairing.

OFAT levels are the existing shipped levels:

| Control | Levels |
|---|---|
| A0 | .1,.25,.5,.75,1 |
| r0 | 0,.25,.5,.75,1 |
| alpha | .5,.7,.9,.95,1 |
| gamma | .1,.5,.9,2,5 |
| f_min | 0,.25,.5,1,1.5 |
| f_max | .5,1,2,5,10 |

Reference appears once, giving 1+6*(5-1)=25 BA configurations. Three datasets x10 seeds
means750 BA runs/2250000 evaluations, plus30 K-Means runs/300 restarts:780 records.
All other controls remain C0. OFAT cannot identify arbitrary cross-factor interactions.

## 5. Two prespecified 3x3 grids

GridG1, loudness coupling:
`A0 in {.25,.5,1}` x `alpha in {.7,.9,1}`; other controls remain C0. A0 and alpha affect
both the acceptance gate and the mean-loudness local step; testing their joint settings
separates a small specified interaction question from six independent OFAT stories.
The reference (.5,.9) is included, and all levels are already present in OFAT.

GridG2, frequency interval:
`f_min in {0,.5,1}` x `f_max in {2,5,10}`; other controls remain C0. This tests joint
location/width of the sampled frequency interval under the known scarce-global-acceptance
mechanism, without choosing the pair after seeing formal frequency findings. The
reference (0,2) is included. Every cell has f_min<f_max; avoiding f_max=.5 in this grid
prevents invalid cells with f_min>=1. Include evaluated/accepted global counts so an
unchanged SSE pattern is not interpreted as an untested or ignored frequency parameter.

Each grid has9 cells: reference +4 single-factor-axis cells already in OFAT +4 new
joint cells. Each therefore contributes only4 new BA configurations to the candidate
pool. Across OFAT and both grids the union is25+4+4=33 unique BA configurations,
including C0. Grid pair/levels must be accepted before discovery; no post-hoc replacement
with r0/gamma because another pair happens to look more significant.

Use cell heatmaps and simple difference-in-differences descriptions for the grids.
Formal interaction significance is not implemented and is not a confirmatory claim of
this proposal. A later factorial model/interaction test requires its own frozen model,
error structure, correction family and tests before inspecting its target outcomes.

## 6. Actual run counts, duplication and execution choice

The current runner deduplicates reference within a single spec only. It does NOT share
completed conditions across separate experiment IDs. Therefore the executable-with-current-
runner plan is OFAT with BA+K-Means, then G1 and G2 with algorithms=[ba] (K-Means is not
repeated for each grid):

| Discovery component | BA runs | BA SSE evaluations | K-Means runs | Raw records |
|---|---:|---:|---:|---:|
| OFAT,25 cells x3 datasets x10 seeds | 750 | 2250000 | 30 | 780 |
| G1,9 cells x3 x10 | 270 | 810000 | 0 | 270 |
| G2,9 cells x3 x10 | 270 | 810000 | 0 | 270 |
| Actual stock-runner discovery total | 1290 | 3870000 | 30 | 1320 |
| Unique configuration/data/seed triples | 990 | 2970000 | 30 | 1020 |

The300 duplicate BA runs (150 per grid) cost900000 extra evaluations. They are repeated
computations of the same seeded configuration, NOT additional independent repetitions.
For analysis select one record per full configuration/dataset/seed with frozen precedence
OFAT >G1 >G2. Before discarding analytic duplicates, require exact agreement of SSE and
nonvolatile histories; a mismatch triggers investigation, not averaging. Retain all raw
records and runtime metadata. Labels such as `A0=.25|alpha=.9` may differ from OFAT labels
while representing the same full config; deduplicate by canonical complete config plus
dataset hash and seed, not textual condition_id.

If the team instead wants the reduced2970000-evaluation discovery budget, it must first
implement and test an explicit cross-experiment reuse registry and artifact references,
then freeze that alternative execution plan. It is not present today. Do not claim the
reduced count while launching all three stock specs. Do not treat resuming one directory
as cross-ID deduplication, and never alter historical pilot directories to simulate reuse.

## 7. Candidate selection and tie handling

Select ONE common configuration from the33 tested configurations; do not construct a new
six-factor combination from six individually best OFAT levels. A configuration outside
this pool requires additional discovery and a new declared budget before confirmation.

Proposed decision rule (thresholds require researcher acceptance):

1. Require all three datasets x10 seeds for a candidate and C0. Any algorithm/numerical
   failure makes a candidate ineligible; report it rather than silently dropping failed
   pairs. An interrupted run may be resumed only after preserving attempts and verifying
   identical source/environment/data. A code defect fix invalidates the affected freeze;
   do not mix code versions or replace a failed seed with a favorable new seed.
2. Rank by lowest Score(c), with equal dataset weights as defined above.
3. For selection eligibility, require at least **1% relative reduction** in Score compared
   with C0 and no dataset's mean Q worse than C0 by more than **2%**. These1%/2% values
   are proposed practical thresholds, not established standards, accepted requirements or
   evidence-derived recommendations. Discuss them before freezing. A failure to pass the
   rule means retain C0, not choose a different endpoint after the fact.
4. Numerical ties: Scores within `1e-6 * max(Score(C0), tiny_positive_fixed_constant)`
   are tied. The constant and tolerance must be frozen, e.g.1e-12 only for numerical
   comparison, never for the Q denominator. Prefer C0 whenever eligible/tied. Among
   nonreference ties, prefer fewer parameters changed from C0; then smaller sum of
   absolute deviations divided by each factor's frozen OFAT range; then lexicographic
   parameter tuple(A0,r0,alpha,gamma,f_min,f_max). Record the full tie-break trace.
5. Do not use discovery p-values, Silhouette, ARI, runtime or pilot confirmation-like
   subsets to override that frozen ranking. Report such outcomes as secondary context.

Archive the selected complete configuration, rule, all eligible scores/failures/ties,
selection timestamp and artifact hashes before opening confirmation. Selection optimizes
noisy discovery means; independent confirmation addresses selection bias but cannot
remove dataset-selection bias or guarantee adequate power.

## 8. Statistical analysis and correction families

Report each dataset/condition's raw values, mean, sample SD(ddof1), median/IQR/range,
nominal95% Student-t mean CI, undefined metrics and failed-run counts. Report nominal95%
CIs for paired Q differences separately; these are not multiplicity-adjusted intervals.
Assumptions and small samples limit interval interpretation. Do not read overlap/nonoverlap
of separate mean CIs as a paired test. Wilcoxon uses paired differences and its usual
location/symmetry interpretation; report ties/zero differences and p-value method/library
version. It is not automatically a test of mean SSE or equivalence.

Discovery is screening: compute32 nonreference-vs-C0 paired tests per dataset, with ten
shared seeds. The proposed exploratory family is all32x3=96 comparisons, Holm-adjusted
as ONE family at.05, not three or multiple experiment-specific families. Three K-Means-
vs-C0 discovery comparisons form a separately labelled secondary family of3. Selection
follows Section7 rather than significance filtering. No interaction-test family is claimed.

Confirmation: one frozen candidate C1, original reference C0, exactly20 unseen seeds.
Prespecify four primary paired comparisons: the aggregate seed-block endpoint Z(C1,s)-
Z(C0,s), plus Q(C1,d,s)-Q(C0,d,s) on each of the three datasets. Use two-sided Wilcoxon
with a Holm family of4 at.05. A negative effect favors C1. The overall descriptive effect
is100*(Score(C0)-Score(C1))/Score(C0), now computed from confirmation only. Proposed
criterion for a confirmed improvement on this fixed three-dataset benchmark: aggregate
Holm-adjusted p<=.05, negative paired aggregate location, at least1% lower aggregate mean
Q, and no dataset's mean Q deterioration>2%. Dataset-specific statistical claims require
that dataset's own adjusted comparison to reject with favorable direction. An unfavorable
or underpowered result is reported honestly; do not retune or open new confirmation seeds.

C1-vs-K-Means and C0-vs-K-Means quality comparisons on three datasets form one separate
secondary Holm family of6. They do not make either algorithm budget-matched. Other metrics
remain descriptive. Report all planned tests including null/unfavorable results and the
prespecified distinction between primary and secondary families. No all-datasets/global
population superiority is inferred from three chosen datasets.

These corrections/normalization/union-selection analyses are NOT implemented by today's
CSV exporter. Current paired_table adjusts only per dataset, per supplied experiment,
and includes every nonreference condition including K-Means. A separately reviewed
analysis script with canonical deduplication, complete-seed checks, Q/Z calculations,
predeclared Holm families and hand-worked tests is a prerequisite to formal analysis;
existing CSVs can be used as raw inputs but cannot be relabelled as this protocol's output.

## 9. Confirmation retains the ORIGINAL reference

Current `confirmation` YAML is a baseline template evaluating only C0+K-Means. Changing
its ba_overrides to C1 would instead evaluate C1 under condition_id='reference'; it would
not retain C0. Do not run that template and report a tuned-vs-original-reference result.

A backward-compatible execution design uses TWO separately frozen explicit experiment
IDs: `confirmation_original_reference` (baseline, ba_base=C0, BA+K-Means) and
`confirmation_frozen_candidate` (baseline, ba_base=C1, BA only), both datasets/seeds
identical. Register roles original_reference/frozen_candidate in an immutable phase
manifest and join by dataset hash+seed in the formal analysis. Both raw engine records
may have condition_id='reference'; use the explicit role/full config/experiment mapping,
not that string alone. Never use either individual auto-generated paired CSV as evidence
for C1-vs-C0. This requires new reviewed definitions/analysis, not a change to BA semantics.

If C1 differs from C0: three datasets x20 seeds x2 BA conditions =120 BA runs/360000
SSE evaluations, plus60 K-Means runs/600 restarts, giving180 confirmation records.
If Section7 retains C0: do not execute a duplicate 'candidate'; run only60 BA/180000
evaluations and60 K-Means/600 restarts(120 records) for benchmark replication, and make
no tuning-gain claim. Reusing the same record as two independent conditions is forbidden.

With a distinct C1, combined formal stock-runner budget is1410 BA runs/4230000 SSE
evaluations,90 K-Means runs/900 restarts and1500 raw records(discovery1320+confirmation180).
A separately implemented/frozen deduplicated discovery alternative would total1110 BA
runs/3330000 evaluations,90 K-Means runs and1200 unique records. These are objective-count
budgets, not runtime estimates. No formal objective evaluation has been performed here.

## 10. Freeze, provenance and explicit go/no-go gates

Before discovery: researcher acceptance of this proposal's thresholds/weights/grids/test
families/failure policy; complete working definitions/analysis tests; exact source commit
and clean tree; pinned environment/build dependencies; dataset/label hashes and feature
order; threadpool/hardware record; stage seed lists; run/cost inventory and output paths.
Publish/archive a protocol SHA256 and immutable phase manifest. Source freeze means the
actual reviewed commit after necessary verification/documentation work, not an assumed
older snapshot. Stage labels and lock rules are not runtime-enforced today.

Before confirmation: freeze candidate-selection evidence, C0/C1 full configs and role
mapping, both confirmation spec hashes, data/source/environment identity and planned
comparisons; verify confirmation outcomes have remained unseen. Obtain explicit approval
to execute each formal phase. Writing this document, passing tests, publishing source,
or running an engineering notebook does not authorize formal execution.

Always use new output IDs/directories. Preserve raw runs, failures, interrupted logs,
original environment/start metadata and hash inventory before any resume. Current resume
checks only spec/data and rewrites some provenance; it does not detect code/environment
changes and does not retain failed attempts. Use an external archive/identity gate or
first implement reviewed provenance preservation; do not silently rely on resume alone.
A protocol deviation requires a dated record before proceeding, never retroactive editing
of the frozen protocol to fit observations.

## 11. Read-only implementation/documentation findings

Line references below describe snapshot 19f20b2; subsequent documentation corrections
may move Markdown line numbers. Python module paths are under src/ba_clustering/.

- `configs/experiments.yaml:68-74` and `experiments.py:250-262`: confirmation template
  has no candidate+original-reference plan. This limitation is correctly acknowledged in
  EXPERIMENT_DESIGN.md:55-60, but must be resolved in formal definitions/role-aware analysis.
- `experiments.py:415-418,444-455`: caching is per experiment directory/run_id, so the
  union pool budget is smaller than the actual three-spec budget. No cross-ID reuse exists.
- `results.py:242-275`, especially272: Holm is currently per dataset and experiment,
  whereas this proposed96-test discovery/four-test primary confirmation family is not.
- `metrics.py:79-108,129-155`: summaries are nominal t intervals and pair tests are
  Wilcoxon; no normalized Q/Z endpoint, candidate selection, adjusted interval or formal
  interaction analysis exists. These are proposed analysis additions, not existing features.
- `results.py:74-93`: environment metadata omits threadpool library/thread counts,
  environment variables, source dirty state and complete dependency lock; the phase
  freeze needs supplemental provenance. `experiments.py:281-285` hashes processed X only,
  not external labels; preserve labels separately.
- `results.py:171-182` and `experiments.py:420-439`: resume rewrites kept JSONL and
  manifest, loses failed-attempt history and refreshes environment metadata. Formal
  immutable provenance cannot rely on this path alone.
- At the start of this continuation, canonical-clone documentation was stale about location/Git: AGENTS.md:15 claims no Git
  repo; HANDOFF.md:3-5 says no commit and names the original workspace; results.git_commit
  at results.py:59-71 now returns a real source commit in this clone. Those historical statements
  are now scoped to the original audit and the canonical clone is recorded separately in
  the updated HANDOFF/AGENTS/publication status.
- ALGORITHM_SPEC.md:327's claim that r0=1 means local search starts only after acceptance
  is mathematically correct; a failed loudness/improvement gate can thus suppress local
  search. ALGORITHM_SPEC.md:321-325's shared-seed trajectory claim applies only when BOTH
  compared settings accept no global candidate; velocity/clipping diagnostics can differ.
  No equation defect was found in the reviewed update path(bat_algorithm.py:209-243).

Review scope: code/document consistency and protocol preparation only. No source,
configuration, historical output or canonical repository file was modified by this review;
the independent reviewer wrote only a draft in /tmp. The parent agent subsequently
added this reviewed proposal to docs/; no formal execution is authorized.


## 12. Repetition-count review before freeze

Ten discovery and twenty confirmation repetitions are the existing roster sizes, not a
power guarantee. Before freeze, use already observed pilot variability to assess achievable
precision/power for the proposed practical threshold and document the calculation. Any
change to repetition counts/seed rosters or budget must be accepted and preregistered
before inspecting the relevant formal outcomes. Do not append favorable seeds or increase
confirmation sample size in response to a disappointing p-value. If the fixed plan is
underpowered, report that limitation or redesign before execution. The proposed 2% per-
dataset guard means relative deterioration of mean Q versus C0, not two percentage points
on the raw Q scale. These proposed thresholds still require researcher acceptance.
