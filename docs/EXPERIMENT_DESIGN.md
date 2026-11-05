# Experiment design

## Research questions and variables

How does each of A0, r0, alpha, gamma, f_min and f_max affect final SSE, across-seed
stability, progress per evaluation, acceptance branches and runtime? Which interactions
matter? Do pilot patterns replicate on unseen algorithm seeds and additional datasets?
Frequency can only alter evaluated global proposals directly; persistent velocity retains
its history even when local proposals replace those moves. Quantify that mechanism.

Independent variables are the six controls. Hold K, processed data, initialization,
boundary/acceptance conventions, population and evaluation budget fixed. Default P=30,
T=99 gives 3000 SSE evaluations including initialization. Disable patience for comparisons.
Reference: A0=r0=0.5, alpha=gamma=0.9, f_min=0, f_max=2; it is not an optimum.
K-Means uses Lloyd, k-means++ initialization, 10 explicit restarts, max_iter=300,
tol=0.0001. Both algorithms use exactly the same processed data and fixed K.

## Data and seeds

Iris (150x4, K=3), Wine (178x13, K=3), synthetic Gaussian blobs (300x4, K=4,
std=1.5, data seed 42), all z-scored once. External classes are evaluation only.
These three datasets do not justify broad claims about all clustering problems.
Future generalization datasets must be chosen before confirmation results are seen;
a new synthetic data seed tests dataset variation, unlike repeating algorithm seeds.

| Stage | Algorithm seeds | Intended use | Executed state |
|---|---|---|---|
| Smoke | 0,1 | Fast engineering checks | Saved smoke plus takeover diagnostics |
| Pilot | 9000..9009 | Feasibility, exploratory levels | Saved baseline/OFAT/grid; selected reruns verified |
| Discovery | 1000..1009 | Formal screening, >=10 paired repetitions | Configured; not run in takeover |
| Confirmation | 2000..2019 | Frozen candidate vs reference and K-Means | Config template only; not run in takeover |

Rosters are disjoint in shipped config and tested. Runtime does not enforce stage locks.
Pair seeds across settings; fixed random consumption keeps initial populations and draws
aligned. Never use reserved formal seeds for debugging. Reusing pilot seeds is allowed,
but those observations cannot become independent confirmation.

## Levels and designs

| Control | Shipped OFAT levels |
|---|---|
| A0 | 0.1,0.25,0.5,0.75,1 |
| r0 | 0,0.25,0.5,0.75,1 |
| alpha | 0.5,0.7,0.9,0.95,1 |
| gamma | 0.1,0.5,0.9,2,5 |
| f_min | 0,0.25,0.5,1,1.5 |
| f_max | 0.5,1,2,5,10 |

OFAT changes one control and includes the reference once. It gives 25 BA conditions;
with K-Means, 3 datasets and 10 seeds: 780 records. It cannot estimate interactions.
Grid expansion takes Cartesian products and includes the reference. The existing 3x3
A0/r0 pilot is a placeholder demonstration, not an evidence-selected pair.
Before formal work, freeze two scientifically motivated 3x3 pair grids, the ranking rule,
primary metric, practical effect threshold, statistical family and computational budget.
Keep f_min<=f_max in all grid cells. The current confirmation template evaluates the
reference only; copy to a new ID with frozen candidate values rather than treating it as
an already configured tuned-vs-reference comparison. Preserve a distinct reference run
for paired comparison; baseline ba_overrides redefine the reference and do not by
themselves retain the original reference. Use a grid or separate frozen experiment and
an explicitly matched analysis for candidate and original reference.

## Evaluation and statistics

Primary endpoint: final SSE within each processed dataset. Secondary: Silhouette,
external ARI, empty-cluster count, runtime, evaluations-to-plateau, accepted local/global
counts, clipping and final max velocity. Keep dataset scales separate; do not pool raw SSE.
Report mean, sample SD, median, IQR, range and 95% Student-t CI on the mean. CI assumptions
and small repetition counts limit inference. Runtime includes fitting, excludes loading,
preprocessing, evaluation metrics and export; control threads and environment.

Paired comparisons use matched seeds, signed SSE differences (candidate-reference), win/
loss/tie counts and two-sided Wilcoxon signed-rank tests. Current code applies Holm per
dataset across all nonreference conditions including K-Means, as one family. Freeze whether
that family matches the formal hypotheses before discovery. Wilcoxon assumptions include
independent paired differences and an appropriate symmetric-difference interpretation;
small/tied samples may limit resolution. Report effect magnitudes and uncertainty, not
only p-values. A non-significant result is not equivalence. No cross-dataset correction or
interaction significance test is currently implemented.

## Discovery and confirmation gates

Before discovery: resolve mechanism concerns without changing the baseline, freeze
protocol/config/data/software snapshot and resource budget. Discovery may select one
candidate using the predefined aggregate rule; record all selection steps. Before opening
confirmation: freeze the candidate and practical threshold, then run only the final
prespecified comparisons. Retain raw results, failures and deviations. Confirmation on
unseen seeds for the same datasets addresses stochastic robustness, not generalization
to unseen datasets. No formal run is authorized by a passing smoke test.


## Formal proposal awaiting freeze

[FORMAL_PROTOCOL_PROPOSAL](FORMAL_PROTOCOL_PROPOSAL.md) specifies a concrete draft:
SSE/TSS analysis endpoint with equal dataset weighting, 25 OFAT conditions plus prespecified
A0/alpha and f_min/f_max grids (33 unique BA configs), paired discovery/confirmation,
selection/tie/failure rules and explicit multiplicity families. Stock-runner discovery
costs 3.87M BA evaluations because cross-experiment deduplication is absent; distinct-
candidate confirmation adds 360k (4.23M total), with K-Means costs reported separately.
Formal normalization/selection/correction analysis and role-aware confirmation definitions
are NOT implemented/frozen. Proposed 1%/2% thresholds need review and power assessment.
No discovery or confirmation seed was executed. Publication or a passing notebook does
not authorize running a formal phase.
