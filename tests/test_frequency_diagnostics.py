from __future__ import annotations

import numpy as np
from scripts.audit_frequency import diagnose

from tests.conftest import make_config


def test_diagnostics_account_for_candidates_and_preserve_production_trace() -> None:
    X = np.array([[1.0], [2.0], [8.0], [9.0]])
    config = make_config()
    result = diagnose(X, config)
    assert result["global_generated"] + result["local_generated"] == (
        config.population_size * config.max_iterations
    )
    for branch in ("global", "local"):
        assert result[f"{branch}_improving"] == (
            result[f"{branch}_accepted"] + result[f"{branch}_gate_rejected"]
        )
    assert result["n_evaluations"] == config.max_evaluations


def test_zero_frequency_disables_all_global_improvements() -> None:
    X = np.array([[1.0], [2.0], [8.0], [9.0]])
    result = diagnose(X, make_config(f_min=0.0, f_max=0.0))
    assert result["global_improving"] == result["global_accepted"] == 0
