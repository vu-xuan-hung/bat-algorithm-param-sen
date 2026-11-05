# Project overview

The research question is how six controls of the specified continuous BA affect
centroid-based clustering quality, stability and computational behavior: A0, r0,
alpha, gamma, f_min and f_max. The objective remains total SSE with fixed K; all
other metrics are evaluation only. The normative implementation choices are in
[ALGORITHM_SPEC](ALGORITHM_SPEC.md), not inferred from ambiguous literature prose.

Current implementation includes validated data/configuration, bounded centroid encoding,
continuous BA, multi-start Lloyd K-Means, preprocessing, baseline/OFAT/grid experiments,
seed repetitions, append-on-completion JSONL, CSV summaries, comparisons, resume and plots.
It contains no hybrid refinement or alternative optimizer. Pilot results exist; their
acceptance mechanisms are diagnostic observations, not confirmed parameter rankings.

Controlled research requires the same processed matrix, K, initialization method, budgets
and paired algorithm seeds. BA objective evaluations and Lloyd iterations are different
cost units. Neither low SSE nor a stalled trace proves the global optimum or agreement
with external classes. See [EXPERIMENT_DESIGN](EXPERIMENT_DESIGN.md).

The takeover verified existing code and preserved previous work. Read
[HANDOFF](HANDOFF.md) for actual checks, remaining limitations and next tasks.
Literature claims inherited in older documents have not been independently source-checked
in this takeover. Parent DOCX/XLSM/PDF archive contents were left unchanged.
