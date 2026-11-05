# Mathematical and scientific audit — 2026-11-05

## Audited specification

The production equations match the requested baseline. Frequency uses one beta per bat,
U[0,1); velocity is v+(position-best)*f; candidate uses the updated velocity. A pulse draw
strictly above current r replaces it with best+epsilon*mean(A); epsilon is independent
U[-1,1) per coordinate. Mean A is snapshotted at sweep start. Candidate clipping precedes
SSE. Strict improvement over the bat's own fitness AND loudness draw<A is required.
A becomes alpha*A and r becomes r0*(1-exp(-gamma*t)) only on acceptance, t starts at 1.
Initial operational r=r0. Rejected proposals retain velocity but leave accepted position,
fitness,A,r unchanged. Accepted improvements update best immediately. All draws are made
up front in fixed order; histories include initialization. Full cost P*(T+1).

SSE is sum_n min_k ||x_n-centroid_k||^2, with no repair or penalty for empty clusters.
Nearest-centroid ties go to the first row. The three requested toy values are 2,1,5.
Velocity for position (2,8), best (1.5,8.5), v=0,f=1 is (0.5,-0.5), hence proposal
(2.5,7.5). Existing tests executed and pass for all those examples.

## Frequency mechanism: ten questions

1. Is rare global acceptance present? Yes in saved pilots and fresh diagnostics. OFAT
   global accepted/evaluated=230/1100140; local accepted/evaluated=161694/1127360.
2. Does the specified sign explain it? The added displacement is away from best for
   positive frequency, exactly as coded and tested. Persistent rejected velocities and
   clipping can exacerbate poor proposals. This is a mechanistic explanation; no
   sign-reversal experiment was conducted, so its isolated causal effect is unmeasured.
3. Is acceptance responsible? Strict improvement is the main observed bottleneck. Across
   fresh reference runs only 3/8783 global proposals improve, and all 3 pass the gate.
   Local improving proposals number 4178; 1348 pass and 2830 fail the loudness gate.
4. Does initialization matter? Yes: separately labelled data-point initialization changes
   SSE and acceptance. It does not remove the scarcity (2 global acceptances/8766 proposals
   across six diagnostic runs). It is not a production option or a new baseline.
5. Does local search dominate? Yes: 1348 local vs 3 global acceptances across the six fresh
   reference runs. This is 99.78% local acceptances for that small diagnostic sample.
6. How often generated? A global velocity and raw proposal are computed every bat update;
   only proposals not replaced by local search are evaluated. Fresh reference: 8783 global
   and 9037 local evaluated candidates out of 17820 updates. Initial 180 evaluations are
   separate. Never count replaced global proposals as tested solutions.
7. How often global accepted? Fresh reference: 3/8783=0.0342%; historical OFAT: 0.0209%.
8. How often local accepted? Fresh reference: 1348/9037=14.92%; OFAT: 14.34%.
9. Is frequency ignored by a bug? No ignored-parameter/update-path bug was found.
   Config integration, formula tests, diagnostic trajectories and changed SSE on Iris
   seed 1/synthetic seed 0 demonstrate its route. A legitimate zero-frequency control
   prevents all global improvement from zero initial velocities.
10. Dataset dependence? Wine reference has zero global improvements on both smoke seeds,
    Iris two, synthetic one. Pilot BA-reference global acceptances over ten seeds are
    Iris 2, Wine 4, synthetic 3. The limited cases support setting/seed/dataset dependence,
    not a universal claim about all datasets or frequency parameters.

Fresh reference diagnostics aggregate seeds 0,1 per dataset with P=30,T=99. The complete
24-run diagnostic also evaluates zero frequency, f_max=10, and data-point initialization;
all standard-initialization traces were compared exactly with ordinary BatClustering.
Observer code calls production run_iteration and never changes state or random consumption.
Source: scripts/audit_frequency.py; raw evidence: verification/frequency_diagnostics.json.
The intervention's separate generator preserves iteration draws but changes initialization.

## Interpretation and defects

No baseline equation was changed. The sign alone is not a proof that global proposals
cannot improve: SSE with changing assignments is nonconvex, best may change within a
sweep, velocity accumulates, and centroid rows need not represent corresponding clusters.
Uniform boxes can be poor in high dimensions. Scale affects the absolute local step.
These factors require controlled studies before attributing quantitative effects.

The two confirmed implementation defects were configuration validation defects outside
the BA mathematics: invalid K-Means numeric types/infinite tolerance, and incomplete
preflight allowing later invalid seeds or K-Means settings to create partial output.
Regression cases failed before fixes and pass afterward. No tests were weakened.

Remaining engineering limits include partial nested YAML validation, low-level typed-array
preconditions, no software-version guard on resume, overwritten resume provenance,
no immutable attempt history, no persisted final centroid arrays, no concurrency lock,
and only first-two-factor grid visualization. These are distinguished from mathematical
behavior and are specified as bounded next tasks in HANDOFF.

## Component checklist after audit

| Component | Exists? | Correctness assessment | Executed testing | Action |
|---|---|---|---|---|
| BA core | Yes, before takeover | Matches specified semantics | Equations/state/toy tests | Preserve; quantify scientific limits |
| SSE fitness | Yes | Total squared nearest-centroid sum, pure | Toy/brute force/overflow/empty tests | None confirmed |
| Clustering representation | Yes | (P,K,D) ordered centroid rows | Shape/assignment/state tests | Keep permutation limitation explicit |
| K-Means baseline | Yes | Lloyd/restarts/shared SSE; validation defects fixed | Baseline + new regressions | Freeze cost/thread protocol |
| Parameter sweeps | Yes | OFAT/grid/shared seeds; preflight fixed | Design/config/integration tests | Freeze formal stage gates |
| Results export | Yes | Saved output validates; resume limited | Export/refusal/interrupted resume | Preserve provenance on future resume |
| Unit tests | Yes | Original suite trustworthy for covered cases | Original 340; final 369 total | Add tests only for meaningful fixes |
| Integration tests | Yes | Marker configured | 7 pass separately | No formal acceptance inference |
| Documentation | Partial before | Missing docs created; README approved | Signature/link/command checks | Fresh install still unverified |

The workspace has no usable Git history. No assertion about tracked/uncommitted changes
is possible; all original files were treated as user work. Historical file hashes were
captured before edits and rechecked afterward. Ten selected historical reruns reproduce
metrics/history, not timing. No formal seed roster was executed.
