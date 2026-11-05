#!/usr/bin/env python
"""Small smoke-seed diagnostics using the unchanged production iteration.

Run from the project root: python scripts/audit_frequency.py --output /tmp/ba-audit.json
The output is created exclusively; existing files are never replaced. No formal seeds.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from ba_clustering import BAConfig, BatClustering, compute_bounds
from ba_clustering.bat_algorithm import (
    draw_iteration_randomness,
    initialize_state,
    run_iteration,
)
from ba_clustering.experiments import (
    build_ba_config,
    expand_conditions,
    load_experiment_spec,
    prepare_dataset,
)
from ba_clustering.fitness import sse_unchecked
from ba_clustering.initialization import initialize_positions
from ba_clustering.results import environment_info
from ba_clustering.validation import FloatArray


def diagnose(
    X: FloatArray, config: BAConfig, *, data_initialization: bool = False
) -> dict[str, Any]:
    """Count generated, improving, gate-rejected and accepted candidates by branch.

    The objective observer sees the exact candidate and current state used by production.
    It draws no randomness and changes no state. Data-point initialization is a separately
    labelled diagnostic intervention, never a change to BatClustering.
    """
    if config.patience is not None:
        raise ValueError("Diagnostics require a fixed iteration budget (patience=None).")
    rng = np.random.default_rng(config.seed)
    bounds = compute_bounds(X)
    initial = initialize_positions(rng, config.population_size, config.n_clusters, bounds)
    if data_initialization:
        # Separate generator preserves the iteration random stream of the reference.
        indices = np.random.default_rng(config.seed).integers(
            len(X), size=(config.population_size, config.n_clusters)
        )
        initial = X[indices].copy()
    state = initialize_state(initial, lambda p: sse_unchecked(X, p), config.r0, config.A0)
    counters: Counter[str] = Counter()
    trace = [state.best_fitness]
    i = 0

    def observe(candidate: FloatArray) -> float:
        nonlocal i
        branch = "local" if draws.pulse_draw[i] > state.pulse_rates[i] else "global"
        value = sse_unchecked(X, candidate)
        counters[f"{branch}_generated"] += 1
        if value < state.fitness[i]:
            counters[f"{branch}_improving"] += 1
            if draws.loudness_draw[i] < state.loudness[i]:
                counters[f"{branch}_accepted"] += 1
            else:
                counters[f"{branch}_gate_rejected"] += 1
        i += 1
        return value

    for t in range(1, config.max_iterations + 1):
        draws = draw_iteration_randomness(rng, config.population_size, initial.shape[1:])
        i = 0
        stats = run_iteration(state, t, draws, observe, config, bounds)
        counters["clipped_coordinates"] += stats.n_clipped_coordinates
        trace.append(state.best_fitness)
    # Check instrumentation against the ordinary public API, including the whole trace.
    if not data_initialization:
        reference = BatClustering(config).fit(X)
        if trace != reference.history_.best_fitness:
            raise AssertionError("Instrumentation changed the best-fitness trajectory.")
        np.testing.assert_array_equal(state.best_position, reference.centroids_)
    fields = {
        f"{branch}_{metric}": counters[f"{branch}_{metric}"]
        for branch in ("global", "local")
        for metric in ("generated", "improving", "accepted", "gate_rejected")
    }
    return {
        "config": config.to_dict(),
        "initialization": "data_points_diagnostic" if data_initialization else "uniform_box",
        "initial_sse": trace[0],
        "final_sse": state.best_fitness,
        "best_fitness_trace": trace,
        "n_evaluations": state.n_evaluations,
        "clipped_coordinates": counters["clipped_coordinates"],
        **fields,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Open exclusively BEFORE spending compute; never touch the historical results tree.
    with args.output.open("x") as handle:
        spec = load_experiment_spec("configs/experiments.yaml", "baseline_pilot")
        reference_condition = expand_conditions(spec)[0]
        rows = []
        for name, options in spec.datasets.items():
            prepared = prepare_dataset(options, spec.preprocessing_method)
            for seed in (0, 1):
                config = build_ba_config(
                    spec, reference_condition, prepared.dataset.n_clusters, seed
                )
                for variant in ("reference", "zero_frequency", "high_frequency", "data_points"):
                    changed = config
                    if variant == "zero_frequency":
                        changed = replace(config, f_min=0.0, f_max=0.0)
                    elif variant == "high_frequency":
                        changed = replace(config, f_min=0.0, f_max=10.0)
                    rows.append(
                        {
                            "dataset": name,
                            "variant": variant,
                            "data_sha256": prepared.sha256,
                            **diagnose(
                                prepared.X, changed, data_initialization=variant == "data_points"
                            ),
                        }
                    )
        json.dump(
            {"purpose": "diagnostic_only", "environment": environment_info(), "runs": rows},
            handle,
            indent=2,
            allow_nan=False,
        )
        handle.write("\n")
    print(args.output)


if __name__ == "__main__":
    main()
