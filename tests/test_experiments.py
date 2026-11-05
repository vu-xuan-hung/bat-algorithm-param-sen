from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from ba_clustering.cli import main
from ba_clustering.experiments import (
    ExperimentSpec,
    build_ba_config,
    expand_conditions,
    list_experiments,
    load_base_config,
    load_experiment_spec,
    run_experiment,
    seeds_overlap,
)
from ba_clustering.plots import generate_plots, interaction_matrix
from ba_clustering.results import (
    ResultsExistError,
    ResultsMismatchError,
    load_runs,
    prepare_directory,
    read_manifest,
)
from ba_clustering.validation import ConfigurationError

REPO = Path(__file__).resolve().parent.parent
EXPERIMENTS = REPO / "configs" / "experiments.yaml"
DEFAULT = REPO / "configs" / "default.yaml"


def small_spec(**overrides: Any) -> ExperimentSpec:
    values: dict[str, Any] = {
        "experiment_id": "unit",
        "description": "unit test",
        "design": "ofat",
        "datasets": {"iris": {"loader": "iris", "n_clusters": 3}},
        "preprocessing_method": "zscore",
        "seeds": (0, 1, 2),
        "algorithms": ("ba", "kmeans"),
        "ba_base": {
            "population_size": 4, "max_iterations": 3, "A0": 0.5, "r0": 0.5, "alpha": 0.9,
            "gamma": 0.9, "f_min": 0.0, "f_max": 2.0, "tolerance": 0.0, "patience": None,
        },
        "kmeans": {"n_init": 2, "init": "k-means++", "max_iter": 50, "tol": 0.0001},
        "factors": {"A0": (0.25, 0.5), "gamma": (0.1,)},
    }  # fmt: skip
    return ExperimentSpec(**(values | overrides))


# --- Shipped configuration -------------------------------------------------------


def test_default_config_budget_is_exactly_the_pilot_budget() -> None:
    base = load_base_config(DEFAULT)
    assert base["ba"]["population_size"] * (base["ba"]["max_iterations"] + 1) == 3000


@pytest.mark.parametrize("experiment_id", list(list_experiments(EXPERIMENTS)))
def test_every_shipped_experiment_is_valid(experiment_id: str) -> None:
    spec = load_experiment_spec(EXPERIMENTS, experiment_id)
    for condition in expand_conditions(spec):
        build_ba_config(spec, condition, 3, spec.seeds[0])


def test_seed_rosters_do_not_overlap() -> None:
    rosters = yaml.safe_load(EXPERIMENTS.read_text())["seed_rosters"]
    names = list(rosters)
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            assert seeds_overlap(rosters[first], rosters[second]) == [], (first, second)


def test_shipped_protocol_minimums() -> None:
    discovery = load_experiment_spec(EXPERIMENTS, "ofat_discovery")
    assert len(discovery.seeds) >= 10
    assert set(discovery.factors) == {"A0", "r0", "alpha", "gamma", "f_min", "f_max"}
    assert all(len(levels) >= 3 for levels in discovery.factors.values())
    assert len(load_experiment_spec(EXPERIMENTS, "confirmation").seeds) >= 20


def test_shipped_frequency_levels_keep_f_min_strictly_below_f_max() -> None:
    spec = load_experiment_spec(EXPERIMENTS, "ofat_discovery")
    for condition in expand_conditions(spec):
        config = build_ba_config(spec, condition, 3, 0)
        assert config.f_min < config.f_max


# --- Design expansion ------------------------------------------------------------


def test_ofat_varies_one_parameter_and_runs_the_reference_once() -> None:
    conditions = expand_conditions(small_spec())
    assert [c.condition_id for c in conditions] == ["reference", "A0=0.25", "gamma=0.1"]
    assert all(len(c.overrides) <= 1 for c in conditions)


def test_grid_crosses_levels() -> None:
    spec = small_spec(design="grid", factors={"A0": (0.25, 0.5), "r0": (0.5, 1.0)})
    assert [c.condition_id for c in expand_conditions(spec)] == [
        "reference", "A0=0.25|r0=0.5", "A0=0.25|r0=1", "A0=0.5|r0=1",
    ]  # fmt: skip


def test_baseline_has_only_the_reference() -> None:
    conditions = expand_conditions(small_spec(design="baseline", factors={}))
    assert [c.condition_id for c in conditions] == ["reference"]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"design": "lhs"}, "Unknown design"),
        ({"factors": {"population_size": (10.0,)}}, "Cannot sweep"),
        ({"design": "baseline"}, "must not declare factors"),
        ({"design": "ofat", "factors": {}}, "at least one factor"),
        ({"design": "grid", "factors": {"A0": (0.5,)}}, "at least two factors"),
    ],
)
def test_invalid_designs_are_rejected(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(ConfigurationError, match=message):
        expand_conditions(small_spec(**overrides))


def test_levels_that_collapse_to_one_condition_id_are_rejected() -> None:
    with pytest.raises(ConfigurationError, match="duplicate condition"):
        expand_conditions(small_spec(factors={"A0": (0.1234561, 0.1234562)}))


def test_invalid_level_fails_before_any_output_exists(tmp_path: Path) -> None:
    spec = small_spec(factors={"f_min": (5.0,)})  # above the reference f_max of 2
    with pytest.raises(ConfigurationError, match="f_min must be <= f_max"):
        run_experiment(spec, tmp_path)
    assert not (tmp_path / "unit").exists()


def write_experiments(tmp_path: Path, entry: dict[str, Any]) -> Path:
    (tmp_path / "default.yaml").write_text(DEFAULT.read_text())
    path = tmp_path / "experiments.yaml"
    path.write_text(
        yaml.safe_dump(
            {"base_config": "default.yaml", "seed_rosters": {"s": [0, 1]},
             "experiments": {"e": entry}}
        )
    )  # fmt: skip
    return path


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        ({"design": "baseline", "datasets": ["iris"], "seeds": "nope"}, "unknown seed roster"),
        ({"design": "baseline", "datasets": ["mnist"], "seeds": "s"}, "datasets must be"),
        ({"design": "baseline", "datasets": [], "seeds": "s"}, "datasets must be"),
        ({"design": "baseline", "datasets": ["iris"], "seeds": "s", "typo": 1}, "Unknown key"),
        (
            {"design": "baseline", "datasets": ["iris"], "seeds": "s", "algorithms": ["pso"]},
            "algorithms must be",
        ),
    ],
)
def test_invalid_experiment_entries_are_rejected(
    tmp_path: Path, entry: dict[str, Any], message: str
) -> None:
    with pytest.raises(ConfigurationError, match=message):
        load_experiment_spec(write_experiments(tmp_path, entry), "e")


def test_unknown_experiment_and_missing_file_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="Unknown experiment"):
        load_experiment_spec(EXPERIMENTS, "nope")
    with pytest.raises(ConfigurationError, match="not found"):
        load_experiment_spec(tmp_path / "missing.yaml", "e")


# --- Outputs ---------------------------------------------------------------------


def test_outputs_contain_every_required_field(tmp_path: Path) -> None:
    outcome = run_experiment(small_spec(), tmp_path)
    assert (outcome.n_completed, outcome.n_failed) == (12, 0)
    records = load_runs(outcome.directory)
    assert len(records) == 12
    ba = next(r for r in records if r["algorithm"] == "ba")
    for field in (
        "experiment_id", "dataset", "algorithm", "config", "seed", "sse", "silhouette",
        "runtime_seconds", "n_evaluations", "history", "status", "software_version", "git_commit",
    ):  # fmt: skip
        assert field in ba
    assert ba["history"]["best_fitness"][-1] == ba["sse"]
    assert ba["n_evaluations"] == 4 * 4
    assert ba["config"]["A0"] in (0.25, 0.5)
    assert ba["n_accepted"] == ba["n_accepted_local"] + ba["n_accepted_global"]
    km = next(r for r in records if r["algorithm"] == "kmeans")
    assert km["n_lloyd_iterations"] >= 2
    assert km["history"] is None
    assert km["n_evaluations"] is None


def test_manifest_records_spec_environment_and_provenance(tmp_path: Path) -> None:
    outcome = run_experiment(small_spec(), tmp_path)
    manifest = read_manifest(outcome.directory)
    assert manifest["status"] == "complete"
    assert manifest["n_runs"] == 12
    assert manifest["spec"]["seeds"] == [0, 1, 2]
    assert manifest["environment"]["numpy"]
    dataset = manifest["datasets"]["iris"]
    assert (dataset["n_samples"], dataset["n_features"], dataset["n_clusters"]) == (150, 4, 3)
    assert dataset["preprocessing"]["method"] == "zscore"
    assert dataset["source"]
    assert len(dataset["sha256"]) == 64


def test_tables_are_written_and_consistent(tmp_path: Path) -> None:
    directory = run_experiment(small_spec(), tmp_path).directory
    runs = pd.read_csv(directory / "runs.csv")
    assert len(runs) == 12
    assert "history" not in runs.columns
    assert {"config_A0", "config_f_max", "sse", "seed"} <= set(runs.columns)

    summary = pd.read_csv(directory / "summary.csv")
    row = summary[(summary.condition_id == "reference") & (summary.metric == "sse")].iloc[0]
    values = runs[(runs.condition_id == "reference")].sse
    assert row.n == 3
    assert row["mean"] == pytest.approx(values.mean())
    assert row["std"] == pytest.approx(values.std(ddof=1))

    paired = pd.read_csv(directory / "paired_vs_reference.csv")
    assert set(paired.condition_id) == {"A0=0.25", "gamma=0.1", "kmeans"}
    assert (paired.p_value_holm >= paired.p_value).all()


def test_ba_cost_is_not_reported_as_kmeans_cost(tmp_path: Path) -> None:
    runs = pd.read_csv(run_experiment(small_spec(), tmp_path).directory / "runs.csv")
    assert runs[runs.algorithm == "kmeans"].n_evaluations.isna().all()
    assert runs[runs.algorithm == "ba"].n_lloyd_iterations.isna().all()


def test_results_are_never_overwritten_without_an_option(tmp_path: Path) -> None:
    directory = run_experiment(small_spec(), tmp_path).directory
    before = (directory / "runs.jsonl").read_text()
    with pytest.raises(ResultsExistError, match="already contains results"):
        run_experiment(small_spec(), tmp_path)
    assert (directory / "runs.jsonl").read_text() == before


def test_overwrite_replaces_results_explicitly(tmp_path: Path) -> None:
    directory = run_experiment(small_spec(), tmp_path).directory
    (directory / "stale.txt").write_text("old")
    outcome = run_experiment(small_spec(), tmp_path, overwrite=True)
    assert outcome.n_completed == 12
    assert not (directory / "stale.txt").exists()


def test_overwrite_refuses_a_directory_that_is_not_results(tmp_path: Path) -> None:
    target = tmp_path / "unit"
    target.mkdir()
    (target / "thesis.docx").write_text("precious")
    with pytest.raises(ResultsExistError, match="Refusing to delete"):
        run_experiment(small_spec(), tmp_path, overwrite=True)
    assert (target / "thesis.docx").exists()


def test_overwrite_and_resume_together_are_rejected(tmp_path: Path) -> None:
    (tmp_path / "file").write_text("x")
    with pytest.raises(ValueError, match="either overwrite or resume"):
        prepare_directory(tmp_path, "hash", overwrite=True, resume=True)


def test_resume_runs_only_what_is_missing_and_matches_a_full_run(tmp_path: Path) -> None:
    full = load_runs(run_experiment(small_spec(), tmp_path / "full").directory)

    directory = run_experiment(small_spec(), tmp_path / "partial").directory
    lines = (directory / "runs.jsonl").read_text().splitlines(keepends=True)
    # Simulate a crash: 5 runs kept, plus a truncated line.
    (directory / "runs.jsonl").write_text("".join(lines[:5]) + lines[5][:40])

    outcome = run_experiment(small_spec(), tmp_path / "partial", resume=True)
    assert (outcome.n_skipped, outcome.n_completed) == (5, 7)
    resumed = load_runs(directory)

    def stable(records: list[dict[str, Any]]) -> dict[str, tuple[Any, Any]]:
        return {r["run_id"]: (r["sse"], r["history"]) for r in records}

    assert stable(resumed) == stable(full)


def test_resume_refuses_a_changed_experiment(tmp_path: Path) -> None:
    run_experiment(small_spec(), tmp_path)
    with pytest.raises(ResultsMismatchError, match="different experiment"):
        run_experiment(small_spec(seeds=(0, 1)), tmp_path, resume=True)


def test_resume_needs_a_manifest(tmp_path: Path) -> None:
    (tmp_path / "unit").mkdir()
    (tmp_path / "unit" / "x").write_text("x")
    with pytest.raises(ResultsExistError, match="nothing to resume"):
        run_experiment(small_spec(), tmp_path, resume=True)


def test_failed_run_is_recorded_and_the_experiment_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ba_clustering.experiments as module

    original = module.run_ba_once

    def flaky(prepared: Any, config: Any) -> dict[str, Any]:
        if config.seed == 1:
            raise FloatingPointError("injected numerical failure")
        return original(prepared, config)

    monkeypatch.setattr(module, "run_ba_once", flaky)
    outcome = run_experiment(small_spec(), tmp_path)
    assert outcome.n_failed == 3
    records = load_runs(outcome.directory)
    failed = [r for r in records if r["status"] == "failed"]
    assert len(failed) == 3
    assert "injected" in failed[0]["error"]
    assert failed[0]["sse"] is None
    assert read_manifest(outcome.directory)["status"] == "complete_with_failures"
    summary = pd.read_csv(outcome.directory / "summary.csv")
    reference = summary[(summary.condition_id == "reference") & (summary.metric == "sse")]
    assert reference.iloc[0].n == 2


def test_programming_errors_are_not_swallowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ba_clustering.experiments as module

    def broken(prepared: Any, config: Any) -> dict[str, Any]:
        raise KeyError("a bug")

    monkeypatch.setattr(module, "run_ba_once", broken)
    with pytest.raises(KeyError):
        run_experiment(small_spec(), tmp_path)


def test_records_are_valid_strict_json(tmp_path: Path) -> None:
    directory = run_experiment(small_spec(), tmp_path).directory
    for line in (directory / "runs.jsonl").read_text().splitlines():
        json.loads(line, parse_constant=lambda name: pytest.fail(f"non-JSON constant {name}"))


# --- Plots -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, {"main_effects_iris.png", "convergence_iris_A0.png", "convergence_iris_gamma.png"}),
        ({"design": "baseline", "factors": {}}, {"baseline_iris.png"}),
        (
            {"design": "grid", "factors": {"A0": (0.25, 0.5), "r0": (0.5, 1.0)}},
            {"interaction_iris_A0_r0.png"},
        ),
    ],
)
def test_plots_are_written_separately_from_data(
    tmp_path: Path, overrides: dict[str, Any], expected: set[str]
) -> None:
    directory = run_experiment(small_spec(**overrides), tmp_path).directory
    paths = generate_plots(directory)
    assert {path.name for path in paths} == expected
    assert all(
        path.parent == directory / "figures" and path.stat().st_size > 1000 for path in paths
    )


def test_grid_plot_finds_every_cell_when_factors_are_not_in_alphabetical_order(
    tmp_path: Path,
) -> None:
    # Regression: sorted manifest keys once reversed the factors, so cells were not found.
    spec = small_spec(design="grid", factors={"r0": (0.5, 1.0), "A0": (0.25, 0.5)})
    directory = run_experiment(spec, tmp_path).directory
    manifest = read_manifest(directory)
    assert list(manifest["spec"]["factors"]) == ["r0", "A0"]
    _, _, matrix = interaction_matrix(load_runs(directory), manifest, "iris", ["r0", "A0"])
    assert np.isfinite(matrix).all()


def test_plots_without_results_raise(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        generate_plots(tmp_path)


# --- CLI -------------------------------------------------------------------------


def test_cli_experiment_plot_and_exit_codes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    arguments = [
        "experiment",
        "smoke",
        "--experiments",
        str(EXPERIMENTS),
        "--output",
        str(tmp_path),
    ]
    assert main(arguments) == 0
    assert (tmp_path / "smoke" / "runs.csv").exists()
    assert main(arguments) == 2
    assert "already contains results" in capsys.readouterr().err
    assert main([*arguments, "--resume"]) == 0
    assert main(["plot", str(tmp_path / "smoke")]) == 0
    assert (tmp_path / "smoke" / "figures" / "main_effects_iris.png").exists()


def test_cli_run_prints_json_and_applies_overrides(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        ["run", "--dataset", "iris", "--seed", "1", "--config", str(DEFAULT),
         "--set", "max_iterations=2", "--set", "A0=0.7"]
    )  # fmt: skip
    assert code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["ba"]["config"]["A0"] == 0.7
    assert output["ba"]["n_evaluations"] == 90
    assert output["kmeans"]["sse"] > 0.0


@pytest.mark.parametrize(
    "arguments",
    [
        ["run", "--dataset", "mnist", "--seed", "1", "--config", str(DEFAULT)],
        ["run", "--dataset", "iris", "--seed", "1", "--config", str(DEFAULT), "--set", "A0=2"],
        ["run", "--dataset", "iris", "--seed", "1", "--config", str(DEFAULT), "--set", "bogus=1"],
        ["run", "--dataset", "iris", "--seed", "1", "--config", str(DEFAULT), "--set", "noequals"],
        ["run", "--dataset", "iris", "--seed", "-1", "--config", str(DEFAULT)],
        ["run", "--dataset", "iris", "--seed", "1", "--config", "missing.yaml"],
        ["experiment", "nope", "--experiments", str(EXPERIMENTS)],
        ["plot", "no/such/results"],
    ],
)
def test_cli_invalid_input_exits_2_with_a_message(
    arguments: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(arguments) == 2
    assert capsys.readouterr().err.startswith("error:")


def test_cli_list(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["list", "--experiments", str(EXPERIMENTS)]) == 0
    assert "ofat_discovery" in capsys.readouterr().out


def test_spec_hash_is_sensitive_to_definition_changes(tmp_path: Path) -> None:
    first = run_experiment(small_spec(), tmp_path / "a").directory
    changed = replace(small_spec(), kmeans={**small_spec().kmeans, "n_init": 3})
    second = run_experiment(changed, tmp_path / "b").directory
    assert read_manifest(first)["spec_hash"] != read_manifest(second)["spec_hash"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"seeds": (0, -1)},
        {"kmeans": {"n_init": 0, "init": "k-means++", "max_iter": 50, "tol": 0.0001}},
    ],
)
def test_all_run_configs_are_validated_before_output(
    tmp_path: Path, overrides: dict[str, Any]
) -> None:
    with pytest.raises(ConfigurationError):
        run_experiment(small_spec(**overrides), tmp_path)
    assert not (tmp_path / "unit").exists()
