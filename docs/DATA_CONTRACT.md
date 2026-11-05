# Data contract

## Inputs

Public clustering and fitness functions validate real numeric arrays. X has shape (N,D),
N,D>=1, finite values, converted to a fresh float64 array. Integers are accepted;
booleans, complex/string/object arrays, ragged data, missing values, infinity and empty
axes are rejected. No imputation or categorical encoding occurs. Centroids have shape
(K,D), K>=1, finite coordinates. Fitting requires K<=N. compute_sse independently allows
more centroids than samples; it is a fitness function, not a fitting configuration.

Positions are (P,K,D), fitness/loudness/pulse vectors (P,), bounds (D,). Low-level pure
updates and unchecked distance/SSE functions assume validated floating-point arrays.
run_bat_algorithm checks population-axis shape and finite objective returns; it is not
a complete validator for arbitrary manually constructed states or injected random draws.
Use BatClustering for validated clustering inputs. Extremely large finite coordinates can
still overflow; scale them. Finite inputs do not promise overflow-free arithmetic.

## Loading and preprocessing

Iris: sklearn bundled numeric 150x4, K=3. Wine: bundled 178x13, K=3. Synthetic default:
make_blobs, N=300, D=4, K=4, cluster_std=1.5, data seed 42. CSV requires a header and
numeric features; optional label_column is removed into labels_external. File paths are
resolved relative to the invoking working directory. External labels affect ARI only.
Dataset validates feature content, K bounds and label length; custom Dataset objects must
supply the declared ndarray types and one-dimensional external labels.

Z-score: (X-mean)/population_std, ddof=0, once per dataset. Exactly constant features
use mean=their value and scale=1, becoming zero. 'none' returns a copy. The estimator
does not preprocess itself. Save the preprocessing metadata and apply it to prediction
inputs in the same units. Returned centroids remain in processed feature units.

## Assignment, objective and boundaries

Squared Euclidean nearest centroid; argmin ties select the lowest index. Empty centroids
contribute nothing to the sum and are never moved/repaired by fitness. Report their count.
Bounds are min/max of processed X per feature. Clip every proposal before evaluation,
keep unclipped velocities. Constant features remain pinned. No centroid permutation
alignment occurs: rows are ordered optimization coordinates even though SSE is invariant
to permuting rows.

## Configuration and output

BAConfig is frozen and keyword-only; ranges and defaults are specified in
[ALGORITHM_SPEC](ALGORITHM_SPEC.md). KMeansConfig requires integral non-boolean K,
seed, n_init and max_iter; nonnegative seed, other values>=1; init is k-means++ or random;
tol is a finite float>=0. Unknown BA mapping keys are errors.

YAML top-level and experiment keys are checked, but arbitrary malformed nested YAML is
not comprehensively schema-validated. Stage labels and seed discipline are conventions
in configs/experiments.yaml; runtime does not block a premature confirmation run.
See [RESULTS_GUIDE](RESULTS_GUIDE.md) for file/field semantics and
[REPRODUCIBILITY](REPRODUCIBILITY.md) for hash and resume limitations.
