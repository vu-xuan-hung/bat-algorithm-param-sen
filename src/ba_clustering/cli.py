"""Command-line interface. Run ``ba-clustering --help`` for usage.

Exit codes: 0 success; 1 at least one experiment run failed; 2 invalid input.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from ba_clustering.baseline import KMeansConfig
from ba_clustering.config import BAConfig
from ba_clustering.experiments import (
    list_experiments,
    load_base_config,
    load_experiment_spec,
    prepare_dataset,
    run_ba_once,
    run_experiment,
    run_kmeans_once,
)
from ba_clustering.plots import generate_plots
from ba_clustering.results import ResultsExistError, ResultsMismatchError
from ba_clustering.validation import ConfigurationError, DataValidationError

DEFAULT_CONFIG = "configs/default.yaml"
DEFAULT_EXPERIMENTS = "configs/experiments.yaml"
USER_ERRORS = (
    ConfigurationError,
    DataValidationError,
    ResultsExistError,
    ResultsMismatchError,
    FileNotFoundError,
)


def _parse_overrides(pairs: Sequence[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for pair in pairs:
        name, separator, value = pair.partition("=")
        if not separator or not name:
            raise ConfigurationError(f"--set expects NAME=VALUE, got {pair!r}.")
        overrides[name] = yaml.safe_load(value)
    return overrides


def _command_run(arguments: argparse.Namespace) -> int:
    """Run BA and K-Means once on one dataset and print a JSON summary. Writes no files."""
    base = load_base_config(arguments.config)
    if arguments.dataset not in base["datasets"]:
        raise ConfigurationError(
            f"Unknown dataset {arguments.dataset!r}. Available: {sorted(base['datasets'])}."
        )
    prepared = prepare_dataset(base["datasets"][arguments.dataset], base["preprocessing"]["method"])
    n_clusters = prepared.dataset.n_clusters
    ba_config = BAConfig.from_mapping(
        {
            **base["ba"],
            **_parse_overrides(arguments.set),
            "n_clusters": n_clusters,
            "seed": arguments.seed,
        }
    )
    ba = run_ba_once(prepared, ba_config)
    ba.pop("history")
    kmeans_config = KMeansConfig(n_clusters=n_clusters, seed=arguments.seed, **base["kmeans"])
    kmeans = run_kmeans_once(prepared, kmeans_config)
    summary = {
        "dataset": prepared.dataset.metadata(),
        "preprocessing": prepared.preprocessing.method,
        "ba": {"config": ba_config.to_dict(), **ba},
        "kmeans": {"config": kmeans_config.to_dict(), **kmeans},
    }
    print(json.dumps(summary, indent=2))
    return 0


def _command_list(arguments: argparse.Namespace) -> int:
    for name, description in list_experiments(arguments.experiments).items():
        print(f"{name}: {description}")
    return 0


def _command_experiment(arguments: argparse.Namespace) -> int:
    spec = load_experiment_spec(arguments.experiments, arguments.experiment_id)
    outcome = run_experiment(
        spec,
        arguments.output,
        overwrite=arguments.overwrite,
        resume=arguments.resume,
        progress=(lambda message: print(message, flush=True)) if arguments.verbose else None,
    )
    print(
        f"{outcome.directory}: {outcome.n_completed} completed, {outcome.n_failed} failed, "
        f"{outcome.n_skipped} kept from a previous run."
    )
    return 1 if outcome.n_failed else 0


def _command_plot(arguments: argparse.Namespace) -> int:
    for path in generate_plots(Path(arguments.results_directory)):
        print(path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser."""
    parser = argparse.ArgumentParser(
        prog="ba-clustering", description="Bat Algorithm clustering and parameter sensitivity."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="Run BA and K-Means once and print a JSON summary.")
    run.add_argument("--dataset", required=True, help="A dataset name from the base config.")
    run.add_argument("--seed", type=int, required=True, help="Algorithm seed.")
    run.add_argument("--config", default=DEFAULT_CONFIG, help="Base config path.")
    run.add_argument(
        "--set", action="append", default=[], metavar="NAME=VALUE", help="Override a BA parameter."
    )
    run.set_defaults(handler=_command_run)

    listing = commands.add_parser("list", help="List experiments.")
    listing.add_argument("--experiments", default=DEFAULT_EXPERIMENTS)
    listing.set_defaults(handler=_command_list)

    experiment = commands.add_parser("experiment", help="Run a declared experiment.")
    experiment.add_argument("experiment_id")
    experiment.add_argument("--experiments", default=DEFAULT_EXPERIMENTS)
    experiment.add_argument("--output", default="results", help="Root directory for results.")
    experiment.add_argument("--verbose", action="store_true", help="Print one line per run.")
    existing = experiment.add_mutually_exclusive_group()
    existing.add_argument("--overwrite", action="store_true", help="Delete existing results first.")
    existing.add_argument("--resume", action="store_true", help="Keep completed runs and continue.")
    experiment.set_defaults(handler=_command_experiment)

    plot = commands.add_parser("plot", help="Generate figures from saved results.")
    plot.add_argument("results_directory")
    plot.set_defaults(handler=_command_plot)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return an exit code."""
    arguments = build_parser().parse_args(argv)
    try:
        code: int = arguments.handler(arguments)
    except USER_ERRORS as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return code


if __name__ == "__main__":
    sys.exit(main())
