# Mathematical Notation

One unified notation for code, documents and tests. Quantities are dimensionless, or in the
units of the preprocessed features where stated. With the default z-score preprocessing,
those units are standard deviations.

## Collisions to avoid

Papers reuse these letters. Do not import their usage into this project.

| Trap | In this project |
|---|---|
| `X` versus bat positions | Code `X` is always the **data matrix**, `(N, D)`. Bat positions are `positions` or `state.positions`, written $X_i$ with a bat subscript in equations. |
| `x_i` | $x_n$ is always a **data point**. Many BA papers use $x_i$ for a bat position; here that is $X_i$. Data is indexed by $n$, and bats by $i$. |
| `K` versus `D` | $K$ is the number of **clusters**. $D$ is the number of **features**. A position is `(K, D)`, in that order. The group's documents also write $d$ for $D$. |
| `A0` versus current loudness | `A0` is a fixed parameter. `A_i`, stored in `state.loudness[i]`, changes during a run. |
| `r0` versus operational rate | `r0` is a fixed parameter. `r_i`, stored in `state.pulse_rates[i]`, is what the local-search trigger compares against. |
| Current versus historical best | $X^*$ is the best **so far** across the whole run, not the best of the current population. `best_fitness` is the historical minimum, and `mean_fitness` describes the current population. |
| `f` | $f_i$ is frequency. Fitness is written `fitness` or $F$, never $f(x)$. |
| `t` | The 1-based iteration counter. History row 0 and $t = 0$ both refer to initialisation. |
| `tolerance` versus `tol` | `tolerance` is BA's improvement threshold. `tol` is scikit-learn's K-Means convergence setting. |
| "evaluations" | `n_evaluations` counts full SSE evaluations by BA. It never describes K-Means, which reports `n_lloyd_iterations`. |

## Dimensions and indices

| Symbol | Code | Meaning | Type | Valid range | Used in |
|---|---|---|---|---|---|
| $N$ | `n_samples`, `X.shape[0]` | Number of samples | int | ≥ 1, and ≥ K | Data, bounds |
| $D$ | `n_features`, `X.shape[1]` | Number of features | int | ≥ 1 | All arrays |
| $K$ | `n_clusters` | Number of clusters | int | [1, N] | Encoding, SSE |
| $P$ | `population_size` | Number of bats | int | ≥ 1 | Population arrays |
| $T$ | `max_iterations` | Iteration budget | int | ≥ 1 | Main loop |
| $n$ | (in vectorised code) | Sample index | int | 1…N | SSE |
| $k$ | (label values, 0-based) | Cluster index | int | 1…K | Assignment |
| $i$ | `i` (0-based) | Bat index | int | 1…P | Sweep |
| $d$ | (in vectorised code) | Feature index | int | 1…D | Bounds |
| $t$ | `t` | Iteration counter | int | 1…T; 0 means initialisation | Pulse-rate schedule, history |

## Data

| Symbol | Code | Meaning | Type | Shape | Valid range | Units | Used in |
|---|---|---|---|---|---|---|---|
| $X$ | `X`, `data` | Preprocessed data matrix | float64 array | `(N, D)` | Finite | Features | Everywhere |
| $x_n$ | `X[n]` | One data point | float64 array | `(D,)` | Finite | Features | SSE |
| $lower_d$ | `bounds.lower` | Per-feature minimum | float64 array | `(D,)` | Finite | Features | Clipping, initialisation |
| $upper_d$ | `bounds.upper` | Per-feature maximum | float64 array | `(D,)` | ≥ lower | Features | Clipping, initialisation |
| $c(x_n)$ | `labels`, `labels_` | Nearest-centroid assignment | int array | `(N,)` | 0…K−1 | None | Metrics |
| | `labels_external` | Evaluation-only class labels | array, or None | `(N,)` | Any | None | ARI only; never in optimisation |

## Bat state

Current state lives in `BatState`, in `src/ba_clustering/bat_algorithm.py`.

| Symbol | Code | Meaning | Type | Shape | Valid range | Units | Used in |
|---|---|---|---|---|---|---|---|
| $X_i$ | `state.positions[i]` | Accepted centroids of bat $i$ | float64 array | `(K, D)` | Within bounds | Features | Eq. 3.2, 3.3 |
| $\mu_k$ | `position[k]` | One centroid | float64 array | `(D,)` | Within bounds | Features | SSE |
| $v_i$ | `state.velocities[i]` | Velocity of bat $i$ | float64 array | `(K, D)` | Finite; may exceed the box | Features per iteration | Eq. 3.2, 3.3 |
| $f_i$ | `frequency` | Frequency for this iteration | float | scalar | [f_min, f_max] | None | Eq. 3.1, 3.2 |
| $F_i$ | `state.fitness[i]` | Accepted fitness of bat $i$ | float | scalar | ≥ 0, finite | Features² | Eq. 3.5 |
| $A_i$ | `state.loudness[i]` | Current loudness | float | scalar | [0, A0] | None | Eq. 3.5, 3.6 |
| $r_i$ | `state.pulse_rates[i]` | Current operational pulse rate | float | scalar | [0, r0] | None | Local-search trigger |
| $\overline{A}^{\,t}$ | `mean_loudness` | Mean loudness, snapshotted at start of iteration $t$ | float | scalar | [0, A0] | None | Eq. 3.4 |
| $X^*$ | `state.best_position` | Best accepted position so far | float64 array | `(K, D)` | Within bounds | Features | Eq. 3.2, 3.4 |
| $F^*$ | `state.best_fitness` | SSE of $X^*$; never increases | float | scalar | ≥ 0, finite | Features² | Output |
| | `candidate` | Clipped trial position, not yet accepted | float64 array | `(K, D)` | Within bounds | Features | Eq. 3.5 |
| | `candidate_fitness` | SSE of the candidate | float | scalar | ≥ 0, finite | Features² | Eq. 3.5 |

## Control parameters

Fixed for a run, and held in `BAConfig`.

| Symbol | Code | Meaning | Type | Valid range | Used in |
|---|---|---|---|---|---|
| $A^0$ | `A0` | Initial loudness | float | [0, 1] | Initialisation |
| $r^0$ | `r0` | Reference pulse rate and initial operational rate | float | [0, 1] | Initialisation, Eq. 3.7 |
| $\alpha$ | `alpha` | Loudness reduction coefficient | float | (0, 1] | Eq. 3.6 |
| $\gamma$ | `gamma` | Pulse-rate growth coefficient | float | (0, ∞) | Eq. 3.7 |
| $f_{\min}$ | `f_min` | Minimum frequency | float | [0, f_max] | Eq. 3.1 |
| $f_{\max}$ | `f_max` | Maximum frequency | float | [f_min, ∞) | Eq. 3.1 |
| $\delta$ | `tolerance` | Meaningful-improvement threshold | float | [0, ∞) | Stopping |
| | `patience` | Stalled iterations before stopping | int, or None | ≥ 1 | Stopping |
| | `seed` | Run seed | int | ≥ 0 | Randomness |

## Random variables

Drawn up front each iteration; see
[ALGORITHM_SPEC.md §6](ALGORITHM_SPEC.md#6-random-draws).

| Symbol | Code | Distribution | Shape | Used in |
|---|---|---|---|---|
| $\beta_i$ | `draws.beta[i]` | $U[0, 1)$ | scalar per bat | Eq. 3.1 |
| $\text{rand}_{\text{pulse}}$ | `draws.pulse_draw[i]` | $U[0, 1)$ | scalar per bat | Local-search trigger: draw $> r_i$ |
| $\epsilon$ | `draws.epsilon[i]` | $U[-1, 1)$, independent per coordinate | `(K, D)` | Eq. 3.4 |
| $\text{rand}_{\text{loud}}$ | `draws.loudness_draw[i]` | $U[0, 1)$ | scalar per bat | Eq. 3.5: draw $< A_i$ |

## Mapping to the group's report

| Report | This project |
|---|---|
| $c_{i,k}$, $\mu_k$ | $\mu_k$, `position[k]` |
| $x_p$, $x_i$ (data) | $x_n$ |
| $d$ (features) | $D$ |
| $T_{\max}$ | $T$, `max_iterations` |
| $\delta$ | `tolerance` |
| `rand1`, `rand2` | `pulse_draw`, `loudness_draw` |
| $1 \times (K \cdot D)$ vector | `(K, D)` array; equivalent up to reshaping |
