"""Dataset loading. No function here needs network access.

External class labels are kept in a separate field and are used only for
explicitly identified evaluation metrics. They never reach an optimiser.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.datasets import load_iris, load_wine, make_blobs

from ba_clustering.validation import (
    ConfigurationError,
    DataValidationError,
    FloatArray,
    validate_data_matrix,
)


@dataclass(frozen=True)
class Dataset:
    """Raw, unscaled data plus metadata.

    Attributes:
        name: Dataset identifier.
        X: Feature matrix of shape ``(n_samples, n_features)``.
        n_clusters: The fixed ``K`` to use for this dataset.
        source: Human-readable provenance.
        labels_external: Optional labels for evaluation only.
        generator: Parameters and seed of a synthetic generator, if any.
    """

    name: str
    X: FloatArray
    n_clusters: int
    source: str
    labels_external: NDArray[Any] | None = None
    generator: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        validate_data_matrix(self.X)
        if not 1 <= self.n_clusters <= self.X.shape[0]:
            raise DataValidationError(
                f"Dataset {self.name!r}: n_clusters={self.n_clusters} must be in "
                f"[1, {self.X.shape[0]}]."
            )
        if self.labels_external is not None and len(self.labels_external) != self.X.shape[0]:
            raise DataValidationError(
                f"Dataset {self.name!r}: {len(self.labels_external)} labels for "
                f"{self.X.shape[0]} samples."
            )

    def metadata(self) -> dict[str, Any]:
        """Return a JSON-serialisable description of the dataset."""
        return {
            "name": self.name,
            "n_samples": self.X.shape[0],
            "n_features": self.X.shape[1],
            "n_clusters": self.n_clusters,
            "source": self.source,
            "has_external_labels": self.labels_external is not None,
            "generator": dict(self.generator) if self.generator else None,
        }


def load_iris_dataset(n_clusters: int = 3) -> Dataset:
    """Load Iris (150 samples, 4 features) from the copy bundled with scikit-learn."""
    bunch = load_iris()
    return Dataset(
        name="iris",
        X=np.asarray(bunch.data, dtype=np.float64),
        n_clusters=n_clusters,
        source="sklearn.datasets.load_iris (bundled with scikit-learn)",
        labels_external=np.asarray(bunch.target),
    )


def load_wine_dataset(n_clusters: int = 3) -> Dataset:
    """Load Wine (178 samples, 13 features) from the copy bundled with scikit-learn."""
    bunch = load_wine()
    return Dataset(
        name="wine",
        X=np.asarray(bunch.data, dtype=np.float64),
        n_clusters=n_clusters,
        source="sklearn.datasets.load_wine (bundled with scikit-learn)",
        labels_external=np.asarray(bunch.target),
    )


def make_synthetic_dataset(
    n_clusters: int, n_samples: int, n_features: int, cluster_std: float, seed: int
) -> Dataset:
    """Generate Gaussian blobs.

    ``seed`` controls only the data. It is independent of the algorithm seeds, so all
    runs of an experiment cluster the same points.
    """
    features, labels = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=cluster_std,
        random_state=seed,
    )
    return Dataset(
        name="synthetic",
        X=np.asarray(features, dtype=np.float64),
        n_clusters=n_clusters,
        source="sklearn.datasets.make_blobs",
        labels_external=np.asarray(labels),
        generator={
            "n_samples": n_samples,
            "n_features": n_features,
            "centers": n_clusters,
            "cluster_std": cluster_std,
            "seed": seed,
        },
    )


def load_csv_dataset(
    path: str | Path, n_clusters: int, label_column: str | None = None, name: str | None = None
) -> Dataset:
    """Load a numeric CSV file with a header row.

    Missing values and non-numeric feature columns are rejected, never imputed.

    Args:
        path: CSV file path.
        n_clusters: The fixed ``K``.
        label_column: Column to set aside as evaluation-only labels.
        name: Dataset name; defaults to the file stem.

    Raises:
        FileNotFoundError: If the file does not exist.
        DataValidationError: If the content violates the data contract.
    """
    csv_path = Path(path)
    frame = pd.read_csv(csv_path)
    labels = None
    if label_column is not None:
        if label_column not in frame.columns:
            raise DataValidationError(
                f"label_column {label_column!r} is not in {list(frame.columns)}."
            )
        labels = frame.pop(label_column).to_numpy()
    non_numeric = [
        str(column) for column in frame.columns if not pd.api.types.is_numeric_dtype(frame[column])
    ]
    if non_numeric:
        raise DataValidationError(f"Non-numeric feature column(s) in {csv_path}: {non_numeric}.")
    return Dataset(
        name=name or csv_path.stem,
        X=validate_data_matrix(frame.to_numpy()),
        n_clusters=n_clusters,
        source=str(csv_path),
        labels_external=labels,
    )


def load_dataset(spec: Mapping[str, Any]) -> Dataset:
    """Build a dataset from a config mapping with a ``loader`` key.

    Supported loaders: ``iris``, ``wine``, ``synthetic``, ``csv``. Remaining keys are
    passed to the loader; unknown keys raise :class:`ConfigurationError`.
    """
    options = dict(spec)
    loader = options.pop("loader", None)
    loaders: dict[str, Any] = {
        "iris": load_iris_dataset,
        "wine": load_wine_dataset,
        "synthetic": make_synthetic_dataset,
        "csv": load_csv_dataset,
    }
    if loader not in loaders:
        raise ConfigurationError(
            f"Unknown dataset loader {loader!r}. Supported: {sorted(loaders)}."
        )
    try:
        dataset: Dataset = loaders[loader](**options)
    except TypeError as error:
        raise ConfigurationError(f"Invalid options for loader {loader!r}: {error}") from error
    return dataset
