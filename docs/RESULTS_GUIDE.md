# Results guide

## 1. File layout

results/<experiment_id>/ contains manifest.json (spec/data/environment/status), runs.jsonl
(raw run records, including histories), runs.csv (flattened records), summary.csv
(descriptive statistics), paired_vs_reference.csv (SSE comparisons), and optional figures/.
JSONL is authoritative for histories and restart SSE; CSV excludes history/config/sweep/
restart_sse as nested structures, expands config_* and stores sweep as JSON text.
Runs do not save final centroids or labels, so independent reconstruction requires a rerun.

## 2. Raw fields

Identification: experiment_id, run_id, dataset, algorithm, condition_id, design, sweep,
seed, config, preprocessing, data_sha256, software_version, git_commit, timestamp_utc.
Completed metric fields: status, error, sse, silhouette, ari_external, n_empty_clusters,
runtime_seconds, n_evaluations, n_iterations, stop_reason, evaluations_to_plateau,
n_local_candidates, n_accepted, n_accepted_local, n_accepted_global,
n_clipped_coordinates, max_abs_velocity_final, n_lloyd_iterations, history.
K-Means additionally saves restart_sse; BA-only fields are null for K-Means and conversely.
Failures have status=failed, error and sse=null; other metric keys can be absent.

BA history is a column dictionary: iteration, n_evaluations, best_fitness, mean_fitness,
mean_loudness, mean_pulse_rate, n_local_candidates, n_accepted, n_accepted_local,
n_clipped_coordinates, max_abs_velocity. Row 0 is initialization. Global evaluated
candidates = P*T - sum(n_local_candidates); global accepted = total accepted - local
accepted. A global velocity/proposal is computed for every bat, but local search may
replace it before evaluation. Distinguish these meanings of 'generated'.

## 3. Metrics and convergence

SSE is total squared nearest-centroid distance; lower is better within one fixed data
representation and K. It is not a ground-truth label score. Silhouette uses Euclidean
sample distances; larger is generally better, but it is secondary and null when fewer
than 2 or at least N distinct labels occur. Empty centroids do not invalidate SSE.
ARI evaluates agreement with external labels only. Never optimize these labels.
Best-so-far SSE is monotone; flat traces indicate no accepted improvement within this
budget, not optimality. Plateau is the first recorded evaluation count reaching final SSE
(with default tolerance zero), measured retrospectively at iteration resolution.
BA's objective count includes P initial evaluations; K-Means reports total Lloyd iterations
across restarts. These are different cost units. Runtime is noisy wall-clock fitting time,
not whole-pipeline cost. Compare it only under controlled hardware, threads and versions.

## 4. Summaries and intervals

Groups are dataset/algorithm/condition. Per metric: n, n_undefined, mean, std (ddof=1),
median, q1, q3, minimum, maximum, ci95_low, ci95_high. IQR=q3-q1. Standard deviation
and CI are null for n<2. CI=mean +/- t(.975,n-1)*SD/sqrt(n), an interval for the mean,
not a range of individual outcomes or a confidence interval for a paired difference.
A group with no defined metric values produces no summary row for that metric.
Show raw values and robust summaries when the distribution is skewed or n is small.

## 5. Statistical comparisons

paired_vs_reference.csv matches completed records on shared seeds, per dataset, versus
BA reference. n can be smaller when runs fail. Negative mean/median difference favors
the listed condition. Counts report n_a_lower, n_b_lower, n_tied. Wilcoxon p_value is
two-sided; all-zero differences yield 1. Holm correction is per dataset across all listed
nonreference conditions including K-Means. It does not correct across datasets.
Report effect size, ties, failures, multiplicity family and practical significance.
No automatic candidate selection or interaction inference is implemented.

## 6. Files, plots and interruptions

Default refusal protects nonempty outputs. Explicit overwrite deletes a directory only
if it contains a manifest; this is destructive. Resume checks spec and processed-data
hashes, keeps completed records, drops failed attempts and a truncated last JSONL line,
then rewrites tables and manifest. Preserve a separate snapshot before any historical
resume; it does not preserve attempt history or original environment metadata.
Interior malformed JSON raises. Software changes are not rejected by resume.
Plots regenerate derived PNGs; do not use historical output directories for routine checks.
OFAT plots show mean +/- SD and mean best-SSE traces. Grid plots show mean SSE, not
statistical significance. Baseline plots show seed values and means. Only the first two
grid factors are plotted, so higher-dimensional grids are not fully represented.

## 7. Pilot observations (not conclusions)

Independently checked on 2026-11-05: smoke 8, baseline_pilot 60, ofat_pilot 780,
interaction_pilot 100 completed records, no failed records. Definition/data hashes and
recomputed summaries match. Ten selected pilot reruns matched every returned non-runtime
metric field, including BA histories and K-Means restart SSE. This is sample reproduction,
not a rerun of every pilot record. See verification/pilot_audit.json.

OFAT has 750 BA runs: 161924 accepted candidates, of which 230 were global and 161694
local. Global acceptance share is 0.142% of acceptances; global acceptance probability is
230/1100140=0.0209% of evaluated global candidates. Do not confuse those denominators.

Fresh diagnostics (seeds 0,1; each run P=30,T=99, default z-score) show:

| Dataset | Global evaluated | Global improving | Global accepted | Local evaluated | Local accepted |
|---|---:|---:|---:|---:|---:|
| Iris | 2898 | 2 | 2 | 3042 | 409 |
| Wine | 2921 | 0 | 0 | 3019 | 523 |
| Synthetic | 2964 | 1 | 1 | 2976 | 416 |

Thus rare strict improvements, rather than the loudness gate alone, explain the observed
scarcity of global acceptances at the reference settings. The added velocity term points
away from best; persistent velocity and boundary clipping further affect proposals. This
mechanistic explanation is consistent with code and measurements; this audit did not
estimate the separate causal effect of changing the velocity sign.

Frequency changes reach the velocity and global candidate; no ignored parameter or
unconditional acceptance bug was found. A zero-frequency diagnostic has zero initial
velocity forever and no strictly improving global proposals. f_max=10 changes Iris seed 1
and synthetic seed 0 trajectories relative to reference, proving the frequency route can
matter. Wine ties on both checked seeds. Data-point initialization changes outcomes but
does not eliminate scarce global acceptance. This intervention is diagnostic only;
uniform-box initialization remains the baseline. Seed and dataset matter. High dimensional
uniform boxes, ordered centroid correspondence, scaling, P/T and r0 are plausible drivers;
the audit does not isolate all of them.

When BOTH compared settings accept no global moves, identical shared-seed accepted-state
and best-SSE trajectories follow by induction from equal initial states and local draws.
Velocities, clipping counts and full histories may still differ. One setting having zero
global acceptances alone does not guarantee equality with another setting.

Evidence and configs: verification/frequency_diagnostics.json and
[SCIENTIFIC_AUDIT](SCIENTIFIC_AUDIT.md). No optimal parameters, general frequency
irrelevance, or BA superiority over K-Means is established by these pilots.


Publication scope: the original workspace retains all historical raw pilot and notebook
session directories. The GitHub publication snapshot excludes those bulk generated
results and all private/user datasets; it includes the executed synthetic notebook and
selected audit evidence under docs/verification. A fresh clone will not contain the
historical raw pilot files. Reproduce engineering checks into fresh output directories;
do not present newly generated runs as the original historical evidence.
