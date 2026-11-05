# Algorithm Specification

This is the normative description of what the code does. If this file and
`src/ba_clustering/bat_algorithm.py` disagree, that is a bug: resolve it, and record
any semantic change in [DECISIONS.md](DECISIONS.md). Notation is defined in
[MATHEMATICAL_NOTATION.md](MATHEMATICAL_NOTATION.md).

## 1. Scope

This is the **Standard Continuous Bat Algorithm** applied to centroid optimisation. It
contains no K-Means refinement, convergence factors, binary encoding, or hybrid local
search. Those change the algorithm itself, so their effects could not be attributed to
the six control parameters this project studies.

## 2. Objective and encoding

Given data $x_1, \dots, x_N \in \mathbb{R}^D$ and a fixed number of clusters $K$, a bat's
position is a candidate centroid set:

| Quantity | Shape |
|---|---|
| One bat's position `X_i` or velocity `v_i` | `(K, D)` |
| Population positions or velocities | `(P, K, D)` |
| `fitness`, `loudness`, `pulse_rates` | `(P,)` |
| Data matrix `X` | `(N, D)` |
| Bounds `lower`, `upper` | `(D,)` |

The objective is the total Sum of Squared Errors, to be minimised:

$$SSE = \sum_{n=1}^{N} \min_{1 \le k \le K} \lVert x_n - \mu_k \rVert_2^2$$

A sample is assigned to its nearest centroid by squared Euclidean distance. Ties go to the
lowest centroid index. SSE stays defined when a centroid owns no samples
([ADR-004](DECISIONS.md#adr-004-empty-cluster-policy)).

## 3. Update equations

Superscripts follow the brief: $X_i^t$ is the accepted state *at the start of* iteration
$t$, and iterations are numbered $t = 1, \dots, T$. So $X_i^1$ is the initial population.

| Eq. | Formula | Function |
|---|---|---|
| 3.1 Frequency | $f_i = f_{\min} + (f_{\max} - f_{\min})\beta_i$, $\beta_i \sim U(0,1)$, drawn per bat per iteration | `frequency_update` |
| 3.2 Velocity | $v_i^{t+1} = v_i^t + (X_i^t - X^*) f_i$ | `velocity_update` |
| 3.3 Position | $X_i^{t+1} = X_i^t + v_i^{t+1}$ (a proposal, not yet accepted) | `position_update` |
| 3.4 Local search | $X_{\text{local}} = X^* + \epsilon\,\overline{A}^{\,t}$, when $\text{rand} > r_i$ | `local_search_candidate` |
| 3.5 Acceptance | $\text{fitness}(\text{candidate}) < \text{fitness}_i$ **and** $\text{rand} < A_i$ | `should_accept` |
| 3.6 Loudness | $A_i^{t+1} = \alpha A_i^t$, on acceptance | `loudness_update` |
| 3.7 Pulse rate | $r_i^{t+1} = r^0 (1 - e^{-\gamma t})$, on acceptance | `pulse_rate_update` |

**Velocity sign.** With $f_i > 0$, the term $(X_i - X^*)f_i$ points **away** from the best
solution. This is the specified equation, and it is not a typo in this code
([ADR-001](DECISIONS.md#adr-001-velocity-sign-convention)). Section 13 describes what
follows from it.

**Perturbation.** $\epsilon$ has shape `(K, D)`, with coordinates independently uniform on
$[-1, 1)$. The step size is absolute, not relative to feature range, so it depends on the
data's scale ([ADR-009](DECISIONS.md#adr-009-local-search-perturbation-and-mean-loudness)).

**Mean loudness.** $\overline{A}^{\,t}$ is the population mean, taken once at the start of
iteration $t$. Acceptances during the sweep do not change it until the next iteration.

**Pulse rate.** `r0` plays two distinct roles:

- It is the **reference parameter** $r^0$ in equation 3.7.
- It is also the **initial operational pulse rate**: $r_i = r0$ before a bat's first
  acceptance.

Equation 3.7 equals zero at $t = 0$, so it is not used for initialisation, and the loop
counter passed to it starts at 1. Consequences, all tested:

- The first acceptance at iteration $t$ sets $r_i = r0(1 - e^{-\gamma t})$, which is
  **below** $r0$. The operational rate therefore usually dips at the first acceptance, then
  rises toward $r0$ on later acceptances. It never exceeds $r0$.
- A bat that never accepts keeps $r_i = r0$.

See [ADR-002](DECISIONS.md#adr-002-pulse-rate-initialisation-and-iteration-index).

## 4. Parameters

| Parameter | Valid range | Reference value | Role |
|---|---|---|---|
| `population_size` | integer ≥ 1 | 30 | Bats, $P$ |
| `max_iterations` | integer ≥ 1 | 99 | Budget, $T$ |
| `n_clusters` | 1 ≤ K ≤ N | per dataset | Centroids per bat, $K$ |
| `A0` | [0, 1] | 0.5 | Initial loudness |
| `r0` | [0, 1] | 0.5 | Reference and initial pulse rate |
| `alpha` | (0, 1] | 0.9 | Loudness decay |
| `gamma` | (0, ∞) | 0.9 | Pulse-rate growth |
| `f_min` | [0, ∞), ≤ `f_max` | 0.0 | Minimum frequency |
| `f_max` | ≥ `f_min` | 2.0 | Maximum frequency |
| `seed` | integer ≥ 0 | per run | Random seed |
| `tolerance` | [0, ∞) | 0.0 | Improvement that counts as meaningful |
| `patience` | integer ≥ 1, or null | null | Stalled iterations before stopping |

Reference values are in `configs/default.yaml`. They come from the group's protocol
sheet, are for screening only, and are not claimed to be optimal.

Edge behaviour worth knowing:

- `A0 = 0`: no candidate can ever be accepted, so the best stays at the initial best.
- `A0 = 1`: the loudness gate passes on every draw until the bat's first decay.
- `alpha = 1`: loudness never decays.
- `r0 = 0`: every candidate is a local-search candidate, for the whole run.
- `r0 = 1`: no local search happens until a bat's first acceptance.
- `f_min = f_max`: frequency is constant.
- `population_size = 1`: the bat is always $X^*$, so its velocity stays exactly zero.

## 5. Initialisation

Performed in `initialization.py` and `bat_algorithm.initialize_state`:

1. `rng = numpy.random.default_rng(seed)`.
2. Positions are drawn uniformly per coordinate within the data's bounding box, shape
   `(P, K, D)` ([ADR-007](DECISIONS.md#adr-007-initialisation-method)).
3. Velocities are zero.
4. $A_i = A0$ and $r_i = r0$ for every bat. Neither is randomised per bat.
5. Every bat is evaluated, costing $P$ evaluations.
6. $X^*$ is the first bat with the minimum fitness. It is a copy that owns its memory.

History row 0 records this state, before any iteration has run.

## 6. Random draws

All randomness for one iteration is drawn up front, in this fixed order, by
`draw_iteration_randomness`:

| Order | Name | Shape | Distribution | Used by |
|---|---|---|---|---|
| 1 | `beta` | `(P,)` | $U[0,1)$ | Equation 3.1 |
| 2 | `pulse_draw` | `(P,)` | $U[0,1)$ | Local-search trigger |
| 3 | `epsilon` | `(P, K, D)` | $U[-1,1)$ | Equation 3.4 |
| 4 | `loudness_draw` | `(P,)` | $U[0,1)$ | Equation 3.5 |

Every value is drawn **whether or not it ends up being used**. Runs with the same seed
therefore consume the same stream under any setting of the six control parameters, which
is what makes paired-seed comparison meaningful
([ADR-006](DECISIONS.md#adr-006-random-number-strategy)).

## 7. Iteration order and pseudocode

Bats are processed sequentially by index. An accepted improvement updates $X^*$
immediately, and later bats in the same sweep see the new value.

```text
bounds  ← per-feature min and max of the preprocessed data
state   ← initialise (section 5)                     # P evaluations
record history row 0

for t = 1, ..., T:
    draws ← draw_iteration_randomness()              # section 6
    Ā ← mean(A)                                      # snapshot for this iteration

    for i = 1, ..., P:
        f ← f_min + (f_max − f_min) · beta[i]
        v[i] ← v[i] + (X[i] − X*) · f                # persists even if rejected
        proposal ← X[i] + v[i]

        if pulse_draw[i] > r[i]:
            proposal ← X* + epsilon[i] · Ā           # replaces the global move

        candidate ← clip(proposal, bounds)           # velocity is not changed
        F ← SSE(candidate)                           # exactly 1 evaluation per bat

        if F < fitness[i] and loudness_draw[i] < A[i]:
            X[i] ← candidate
            fitness[i] ← F
            A[i] ← α · A[i]
            r[i] ← r0 · (1 − exp(−γ t))
            if F < best_fitness:
                X* ← copy(candidate)
                best_fitness ← F

    record history row t
    if patience is set and stalled for patience iterations: stop

return X*, best_fitness, history
```

**Cost.** A full run costs exactly $P(1 + T)$ objective evaluations. The reference
$P = 30$, $T = 99$ gives exactly the 3,000-evaluation pilot budget.

## 8. Acceptance and best-so-far tracking

The code keeps these separate:

| Concept | Variable |
|---|---|
| Current accepted position | `state.positions[i]` |
| Proposed candidate position | local `candidate` |
| Current accepted fitness | `state.fitness[i]` |
| Candidate fitness | local `candidate_fitness` |
| Global best-so-far position | `state.best_position` |
| Global best-so-far fitness | `state.best_fitness` |

Rules, each covered by a test:

- A rejected candidate changes only the velocity, which was already updated. Position,
  fitness, loudness and pulse rate do not change.
- A candidate that beats the global best but fails the loudness draw is **discarded**. The
  global best changes only through accepted solutions.
- An accepted candidate that improves on the bat's own fitness but not on the best moves
  the bat, and leaves $X^*$ alone.
- Best-so-far fitness never increases, and it always equals the recomputed SSE of the
  stored best position.
- `fitness[i]` always equals the recomputed SSE of `positions[i]`.

## 9. Boundary handling

Bounds are computed from the **preprocessed** data: $lower_d = \min_n x_{nd}$,
$upper_d = \max_n x_{nd}$.

| Question | Answer |
|---|---|
| What is clipped? | Every candidate, global or local, coordinate-wise, before evaluation. |
| Does clipping change velocity? | **No.** Velocity keeps its unclipped value and can grow beyond the box. |
| Does it affect local search? | Yes, in the same way. A local candidate outside the box is clipped. |
| Zero-range features? | Lower equals upper, so that coordinate is pinned to the constant. `Bounds.zero_range_features` reports them. They add nothing to SSE. |
| Are clips counted? | Yes: `n_clipped_coordinates` per iteration, in the history. |

No other boundary correction exists. See
[ADR-005](DECISIONS.md#adr-005-boundary-policy).

## 10. Stopping

| Mode | Configuration | `stop_reason` |
|---|---|---|
| Fixed budget (default) | `patience: null` | `max_iterations` |
| Early stop | `patience: n`, optional `tolerance` | `patience`, or `max_iterations` if never triggered |

An iteration is **stalled** when best fitness has not improved by more than `tolerance`
since the last meaningful improvement. The comparison is against that reference, not the
previous iteration, so a slow drift of tiny gains cannot keep resetting the counter
indefinitely. A `tolerance > 0` without `patience` is rejected, because it would have no
effect.

A `patience` stop means improvement stalled. It is not evidence of a global optimum.

## 11. Numerical toy example

This is reproduced by `test_documented_toy_iteration` in `tests/test_ba_updates.py`. Keep
them in sync.

**Setup:** $X = \{1, 2, 8, 9\}$, so $K = 2$, $D = 1$, $P = 3$, with bounds $[1, 9]$.
$A0 = r0 = 0.5$, $\alpha = \gamma = 0.9$, $f \in [0, 2]$.

| Bat | Initial centroids | SSE | Note |
|---|---|---|---|
| 1 | (2, 8) | 2.0 | 1 + 0 + 0 + 1 |
| 2 | (1.2, 8.6) | 1.2 | 0.04 + 0.64 + 0.36 + 0.16; this is $X^*$ |
| 3 | (3, 7) | 10.0 | 4 + 1 + 1 + 4 |

**Iteration $t = 1$**, with $\overline{A}^{\,1} = 0.5$ and these injected draws:

| Bat | $\beta$ | `pulse_draw` | $\epsilon$ | `loudness_draw` |
|---|---|---|---|---|
| 1 | 0.5 | 0.2 | (0, 0) | 0.1 |
| 2 | 0.25 | 0.9 | (0.6, −0.2) | 0.3 |
| 3 | 0.75 | 0.7 | (0.2, −0.2) | 0.6 |

**Bat 1, global move, rejected.** $f = 1$. $v = (2 - 1.2,\ 8 - 8.6) \cdot 1 = (0.8, -0.6)$.
Since $0.2 \le r = 0.5$ there is no local search, and the candidate is $(2.8, 7.4)$ with
SSE $3.24 + 0.64 + 0.36 + 2.56 = 6.8$. That is worse than 2.0, so it is rejected. The
position stays at (2, 8), and the velocity stays at (0.8, −0.6).

**Bat 2, local move, accepted, new best.** $f = 0.5$. It is $X^*$, so $v = (0, 0)$. Since
$0.9 > 0.5$, local search gives $(1.2, 8.6) + (0.6, -0.2) \cdot 0.5 = (1.5, 8.5)$, with
SSE $1.0 < 1.2$. Since $0.3 < A_2 = 0.5$, it is accepted:

- $A_2 = 0.9 \cdot 0.5 = 0.45$
- $r_2 = 0.5(1 - e^{-0.9}) \approx 0.29672$, which is below $r0$ (see section 3)
- $X^*$ becomes $(1.5, 8.5)$, with best fitness 1.0

**Bat 3, improving but rejected by loudness.** $f = 1.5$. Using the *updated* best,
$v = (3 - 1.5,\ 7 - 8.5) \cdot 1.5 = (2.25, -2.25)$. Since $0.7 > 0.5$, local search
gives $(1.5, 8.5) + (0.2, -0.2) \cdot 0.5 = (1.6, 8.4)$, with SSE
$0.36 + 0.16 + 0.16 + 0.36 = 1.04$. That improves on 10.0, but $0.6 \not< A_3 = 0.5$, so
it is **rejected**. Position, fitness, loudness and pulse rate are all unchanged.

**After the iteration:**

| Quantity | Value |
|---|---|
| Positions | (2, 8), (1.5, 8.5), (3, 7) |
| Fitness | 2.0, 1.0, 10.0 |
| Loudness | 0.5, 0.45, 0.5 (mean 0.48333) |
| Pulse rates | 0.5, 0.29672, 0.5 |
| Best | (1.5, 8.5), SSE 1.0 |
| Evaluations | 3 + 3 = 6 |

## 12. Ambiguities between the literature and this implementation

The sources disagree or leave things open. Each resolution below is deliberate and is not
a claim about what is correct in general. Sources are the PDFs in the group's papers
archive, except Yang (2010), which **is not in the archive**, so its original text was not
checked here. Verify against it before citing.

| Topic | Literature | This implementation |
|---|---|---|
| Acceptance comparison | Komarasamy & Wahi and Zhu et al. pseudocode: `rand < A_i & f(x_i) < f(x*)`, the **global** best. | The bat's **own** current fitness, per the brief. |
| Local search centre | Carvalho et al. eq. 4: around $x_i$. Others: around a selected best solution. | Around $X^*$, per the brief. |
| Local step size | $\epsilon\overline{A}$ with $\epsilon \in [-1,1]$ in several papers. Fong et al. eq. 9: $x^* + 0.01 \times \text{rand}()$. | $\epsilon\overline{A}$ with vector-valued $\epsilon$. No extra scale factor. |
| Is $\epsilon$ scalar or a vector? | Usually called "a random number". | An independent value per coordinate, per the brief. |
| Pulse-rate schedule | Carvalho et al. eq. 6 is recursive: $r^t = r^{t-1}[1 - e^{-\gamma t}]$. Others use $r^0[1 - e^{-\gamma t}]$. | $r^0[1 - e^{-\gamma t}]$. |
| Meaning of $r^0$ | Komarasamy & Wahi and Carvalho et al.: the initial rate "can be around" zero. | `r0` is the reference and the initial rate, in [0, 1]. |
| Range of $A^0$ | Komarasamy & Wahi and Carvalho et al.: "typically [1, 2]". | [0, 1], per the brief. |
| Parameter initialisation | Carvalho et al. randomise per bat, and tie $\alpha = \gamma$. | Shared values; $\alpha$ and $\gamma$ independent, because they are studied separately. |
| Position update index | Carvalho et al. eq. 3 is printed with $v^{t-1}$. | Uses the just-updated velocity, per the brief. |
| Velocity direction in prose | The group report (§2.2.3) says "toward" the best, and Fong et al. describe a position "to move towards". | The equation's sign is authoritative. With $f > 0$ the update points **away** from $X^*$. |
| Candidates per iteration | Pseudocode lists both a local solution and "a new solution by flying randomly", without stating the evaluation count. | One candidate per bat. A local candidate replaces the global one. |

## 13. Behavioural consequences observed in pilots

These follow from the specification, and they matter for interpreting any sensitivity
result. Measurements and caveats are in
[RESULTS_GUIDE.md](RESULTS_GUIDE.md#7-pilot-observations-not-conclusions).

- **Global moves are almost never accepted.** In the pilot OFAT, 230 of 161,924 accepted
  candidates (0.14%) came from global moves. The repelling sign, plus velocity that keeps
  accumulating on rejected moves, sends proposals outside the box, where they are clipped.
- **`f_min` and `f_max` can only act through global candidates.** Frequency has no other
  route into the algorithm. When neither compared frequency setting accepts a global candidate, accepted-position
  and best-SSE trajectories are *exactly identical* under a shared seed (velocity and
  clipping diagnostics may still differ). The pilot's many tied
  seeds in frequency comparisons are this effect, not a coincidence.
- **Local search does nearly all of the improving,** so `r0` has a strong effect at the
  extreme: at `r0 = 1`, local search does not start until a first acceptance, which rarely
  comes.
- **`A0` and `alpha` act on both the acceptance gate and the step size,** because $A_i$
  gates acceptance and $\overline{A}$ scales local steps. Their effects cannot be separated
  by a single sweep.


## 14. Independent takeover verification (2026-11-05)

All equations, toy examples and documented update conventions were independently checked
against source and executed tests; no equation changed. See SCIENTIFIC_AUDIT.md and
verification evidence for candidate-generation denominators and limitations. The original
literature table is retained as prior documentation; its paper attributions were not
independently source-checked during takeover. At r0=0, local search has probability one
under ideal continuous draws, but the strict comparison remains draw>0; an exactly zero
PRNG draw would select the global branch. Tests pin that strict comparison.
