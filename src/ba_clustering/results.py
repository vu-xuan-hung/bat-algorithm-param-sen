"""Result storage and tabulation. All experiment file I/O lives here.

Layout of ``results/<experiment_id>/``:

* ``manifest.json``: experiment snapshot, environment, and completion status.
* ``runs.jsonl``: one complete record per run, appended as each run finishes.
* ``runs.csv``: the same records, flattened, without per-iteration histories.
* ``summary.csv``: descriptive statistics per dataset, algorithm, condition and metric.
* ``paired_vs_reference.csv``: paired SSE comparisons against the reference condition.
* ``figures/``: plots, written separately by :mod:`ba_clustering.plots`.

See ``docs/RESULTS_GUIDE.md`` for the schemas.
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from ba_clustering.metrics import holm_adjust, paired_comparison, summarize

MANIFEST_NAME = "manifest.json"
RUNS_JSONL_NAME = "runs.jsonl"
SUMMARY_METRICS = (
    "sse",
    "silhouette",
    "ari_external",
    "runtime_seconds",
    "n_evaluations",
    "evaluations_to_plateau",
    "n_empty_clusters",
)
_CSV_EXCLUDED = ("history", "config", "sweep", "restart_sse")


class ResultsExistError(FileExistsError):
    """The output directory already holds results and no explicit option was given."""


class ResultsMismatchError(ValueError):
    """Results being resumed were produced by a different experiment definition."""


def canonical_hash(value: Any) -> str:
    """Return the SHA-256 hex digest of ``value`` as canonical JSON."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def git_commit() -> str | None:
    """Return the current commit of the source tree, or ``None`` outside a git repository."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def environment_info() -> dict[str, Any]:
    """Return the software environment for the manifest."""
    import matplotlib
    import numpy
    import scipy
    import sklearn

    import ba_clustering

    return {
        "ba_clustering": ba_clustering.__version__,
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__,
    }


def _dump(value: Any) -> str:
    return json.dumps(value, allow_nan=False)


def write_manifest(directory: Path, manifest: Mapping[str, Any]) -> None:
    """Write the manifest atomically. Key order is preserved: factor order is meaningful."""
    temporary = directory / f"{MANIFEST_NAME}.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    temporary.replace(directory / MANIFEST_NAME)


def read_manifest(directory: Path) -> dict[str, Any]:
    """Read the manifest of a results directory.

    Raises:
        FileNotFoundError: If the directory has no manifest.
    """
    manifest: dict[str, Any] = json.loads((directory / MANIFEST_NAME).read_text())
    return manifest


def append_run(directory: Path, record: Mapping[str, Any]) -> None:
    """Append one run record to ``runs.jsonl`` and flush it to the OS."""
    with (directory / RUNS_JSONL_NAME).open("a") as handle:
        handle.write(_dump(record) + "\n")
        handle.flush()


def load_runs(directory: Path) -> list[dict[str, Any]]:
    """Load all run records.

    A truncated *last* line is ignored, because an interrupted process can leave one.
    Corruption anywhere else raises, since it means the file was damaged.

    Raises:
        json.JSONDecodeError: If a line other than the last is invalid.
    """
    path = directory / RUNS_JSONL_NAME
    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    records: list[dict[str, Any]] = []
    for number, line in enumerate(lines):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if number != len(lines) - 1:
                raise
    return records


def prepare_directory(
    directory: Path, spec_hash: str, *, overwrite: bool = False, resume: bool = False
) -> list[dict[str, Any]]:
    """Make ``directory`` ready for a run and return completed records to keep.

    Args:
        directory: Target directory.
        spec_hash: Hash of the experiment definition.
        overwrite: Delete existing results first.
        resume: Keep completed runs, provided the experiment definition is unchanged.

    Raises:
        ResultsExistError: If results exist and neither option was given, or if an
            overwrite target is not a results directory.
        ResultsMismatchError: If resuming results from a different definition.
        ValueError: If both options are set.
    """
    if overwrite and resume:
        raise ValueError("Choose either overwrite or resume, not both.")
    has_content = directory.exists() and any(directory.iterdir())
    if not has_content:
        directory.mkdir(parents=True, exist_ok=True)
        return []
    is_results = (directory / MANIFEST_NAME).exists()
    if resume:
        if not is_results:
            raise ResultsExistError(f"{directory} has no {MANIFEST_NAME}; nothing to resume.")
        existing_hash = read_manifest(directory).get("spec_hash")
        if existing_hash != spec_hash:
            raise ResultsMismatchError(
                f"Cannot resume {directory}: it was produced by a different experiment "
                f"definition (hash {existing_hash} versus {spec_hash})."
            )
        kept = [record for record in load_runs(directory) if record["status"] == "completed"]
        (directory / RUNS_JSONL_NAME).write_text("".join(_dump(r) + "\n" for r in kept))
        return kept
    if overwrite:
        if not is_results:
            raise ResultsExistError(
                f"Refusing to delete {directory}: it has no {MANIFEST_NAME}, so it does not "
                "look like a results directory."
            )
        shutil.rmtree(directory)
        directory.mkdir(parents=True)
        return []
    raise ResultsExistError(
        f"{directory} already contains results. Use a new experiment ID, or pass an explicit "
        "--resume or --overwrite."
    )


def flatten_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return a CSV row: scalars kept, config expanded to ``config_*``, sweep as JSON text."""
    row = {key: value for key, value in record.items() if key not in _CSV_EXCLUDED}
    row["sweep"] = _dump(record.get("sweep") or {})
    for key, value in (record.get("config") or {}).items():
        row[f"config_{key}"] = value
    return row


def runs_table(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Return all records as a flat table."""
    return pd.DataFrame([flatten_record(record) for record in records])


def summary_table(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Return descriptive statistics of completed runs, one row per group and metric.

    ``n_undefined`` counts runs where the metric was unavailable, such as a Silhouette
    score that is undefined for a single non-empty cluster.
    """
    rows: list[dict[str, Any]] = []
    completed = [record for record in records if record["status"] == "completed"]
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for record in completed:
        key = (record["dataset"], record["algorithm"], record["condition_id"])
        groups.setdefault(key, []).append(record)
    for (dataset, algorithm, condition), members in sorted(groups.items()):
        for metric in SUMMARY_METRICS:
            values = [m[metric] for m in members if m.get(metric) is not None]
            if not values:
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "algorithm": algorithm,
                    "condition_id": condition,
                    "metric": metric,
                    "n_undefined": len(members) - len(values),
                    **asdict(summarize(values)),
                }
            )
    return pd.DataFrame(rows)


def paired_table(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Compare the SSE of each condition with the BA reference on shared seeds.

    ``a`` is the listed condition and ``b`` is the reference, so a negative difference
    means the condition has lower SSE. Holm adjustment is applied per dataset.
    """
    completed = [record for record in records if record["status"] == "completed"]
    by_dataset: dict[str, dict[str, dict[int, float]]] = {}
    for record in completed:
        conditions = by_dataset.setdefault(record["dataset"], {})
        conditions.setdefault(record["condition_id"], {})[record["seed"]] = record["sse"]

    rows: list[dict[str, Any]] = []
    for dataset, conditions in sorted(by_dataset.items()):
        reference = conditions.get("reference")
        if reference is None:
            continue
        dataset_rows: list[dict[str, Any]] = []
        for condition, values in sorted(conditions.items()):
            if condition == "reference":
                continue
            seeds = sorted(set(values) & set(reference))
            if not seeds:
                continue
            comparison = paired_comparison(
                [values[seed] for seed in seeds], [reference[seed] for seed in seeds]
            )
            dataset_rows.append(
                {"dataset": dataset, "condition_id": condition, **asdict(comparison)}
            )
        adjusted = holm_adjust([row["p_value"] for row in dataset_rows])
        for row, value in zip(dataset_rows, adjusted, strict=True):
            row["p_value_holm"] = value
        rows.extend(dataset_rows)
    return pd.DataFrame(rows)


def write_tables(directory: Path, records: Sequence[Mapping[str, Any]]) -> None:
    """Write ``runs.csv``, ``summary.csv`` and ``paired_vs_reference.csv``."""
    runs_table(records).to_csv(directory / "runs.csv", index=False)
    summary_table(records).to_csv(directory / "summary.csv", index=False)
    paired_table(records).to_csv(directory / "paired_vs_reference.csv", index=False)
