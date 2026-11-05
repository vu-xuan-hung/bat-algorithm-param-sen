"""Standard Continuous Bat Algorithm for centroid-based clustering."""

from ba_clustering.baseline import KMeansConfig, KMeansResult, run_kmeans
from ba_clustering.bat_algorithm import BAResult, ConvergenceHistory, run_bat_algorithm
from ba_clustering.boundaries import Bounds, clip_to_bounds, compute_bounds
from ba_clustering.clustering import (
    BatClustering,
    assign_labels,
    cluster_sizes,
    count_empty_clusters,
)
from ba_clustering.config import BAConfig
from ba_clustering.datasets import (
    Dataset,
    load_csv_dataset,
    load_dataset,
    load_iris_dataset,
    load_wine_dataset,
    make_synthetic_dataset,
)
from ba_clustering.fitness import compute_sse
from ba_clustering.preprocessing import PreprocessingInfo, preprocess
from ba_clustering.validation import ConfigurationError, DataValidationError, NotFittedError

__version__ = "0.1.0"

__all__ = [
    "BAConfig",
    "BAResult",
    "BatClustering",
    "Bounds",
    "ConfigurationError",
    "ConvergenceHistory",
    "DataValidationError",
    "Dataset",
    "KMeansConfig",
    "KMeansResult",
    "NotFittedError",
    "PreprocessingInfo",
    "__version__",
    "assign_labels",
    "clip_to_bounds",
    "cluster_sizes",
    "compute_bounds",
    "compute_sse",
    "count_empty_clusters",
    "load_csv_dataset",
    "load_dataset",
    "load_iris_dataset",
    "load_wine_dataset",
    "make_synthetic_dataset",
    "preprocess",
    "run_bat_algorithm",
    "run_kmeans",
]
