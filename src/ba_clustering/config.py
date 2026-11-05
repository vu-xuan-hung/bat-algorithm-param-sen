"""Strictly validated configuration for the Bat Algorithm.

Reference parameter values live in ``configs/default.yaml``, not here. Only the
stopping options have defaults, and those defaults mean "fixed iteration budget".
"""

from __future__ import annotations

import dataclasses
import math
import numbers
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ba_clustering.validation import ConfigurationError


def _require_int(name: str, value: Any, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ConfigurationError(f"{name} must be an integer, got {value!r}.")
    if value < minimum:
        raise ConfigurationError(f"{name} must be >= {minimum}, got {value}.")


def _require_finite_real(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ConfigurationError(f"{name} must be a real number, got {value!r}.")
    if not math.isfinite(float(value)):
        raise ConfigurationError(f"{name} must be finite, got {value!r}.")


@dataclass(frozen=True, kw_only=True)
class BAConfig:
    """Parameters of one Bat Algorithm clustering run.

    Attributes:
        population_size: Number of bats, ``P >= 1``.
        max_iterations: Iteration budget, ``T >= 1``.
        n_clusters: Number of centroids per bat, ``K >= 1``.
        A0: Initial loudness of every bat, in ``[0, 1]``.
        r0: Reference pulse-rate parameter in ``[0, 1]``; also the initial
            operational pulse rate of every bat. See DECISIONS.md, ADR-002.
        alpha: Loudness reduction coefficient, in ``(0, 1]``.
        gamma: Pulse-rate growth coefficient, ``> 0``.
        f_min: Minimum frequency, ``>= 0``.
        f_max: Maximum frequency, ``>= f_min``.
        seed: Non-negative seed for ``numpy.random.default_rng``.
        tolerance: Best-fitness improvement that counts as meaningful, ``>= 0``.
            Values above zero require ``patience``.
        patience: Consecutive iterations without meaningful improvement before
            stopping, or ``None`` for the fixed budget.
    """

    population_size: int
    max_iterations: int
    n_clusters: int
    A0: float
    r0: float
    alpha: float
    gamma: float
    f_min: float
    f_max: float
    seed: int
    tolerance: float = 0.0
    patience: int | None = None

    def __post_init__(self) -> None:
        _require_int("population_size", self.population_size, 1)
        _require_int("max_iterations", self.max_iterations, 1)
        _require_int("n_clusters", self.n_clusters, 1)
        _require_int("seed", self.seed, 0)

        for name in ("A0", "r0", "alpha", "gamma", "f_min", "f_max", "tolerance"):
            _require_finite_real(name, getattr(self, name))

        if not 0.0 <= self.A0 <= 1.0:
            raise ConfigurationError(f"A0 must be in [0, 1], got {self.A0}.")
        if not 0.0 <= self.r0 <= 1.0:
            raise ConfigurationError(f"r0 must be in [0, 1], got {self.r0}.")
        if not 0.0 < self.alpha <= 1.0:
            raise ConfigurationError(f"alpha must be in (0, 1], got {self.alpha}.")
        if self.gamma <= 0.0:
            raise ConfigurationError(f"gamma must be > 0, got {self.gamma}.")
        if self.f_min < 0.0:
            raise ConfigurationError(f"f_min must be >= 0, got {self.f_min}.")
        if self.f_min > self.f_max:
            raise ConfigurationError(
                f"f_min must be <= f_max, got f_min={self.f_min} and f_max={self.f_max}."
            )
        if self.tolerance < 0.0:
            raise ConfigurationError(f"tolerance must be >= 0, got {self.tolerance}.")

        if self.patience is not None:
            _require_int("patience", self.patience, 1)
        elif self.tolerance > 0.0:
            raise ConfigurationError(
                f"tolerance={self.tolerance} has no effect without patience. "
                "Set patience to an integer >= 1, or set tolerance to 0."
            )

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> BAConfig:
        """Build a config from a mapping, rejecting unknown and missing keys."""
        fields = {field.name: field for field in dataclasses.fields(cls)}
        unknown = sorted(set(values) - set(fields))
        if unknown:
            raise ConfigurationError(
                f"Unsupported configuration option(s): {unknown}. Supported: {sorted(fields)}."
            )
        missing = sorted(
            name
            for name, field in fields.items()
            if field.default is dataclasses.MISSING and name not in values
        )
        if missing:
            raise ConfigurationError(f"Missing required configuration option(s): {missing}.")
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        """Return all fields as a JSON-serialisable dictionary."""
        return dataclasses.asdict(self)

    @property
    def max_evaluations(self) -> int:
        """Objective evaluations in a full run: ``P`` initial plus ``P`` per iteration."""
        return self.population_size * (self.max_iterations + 1)
