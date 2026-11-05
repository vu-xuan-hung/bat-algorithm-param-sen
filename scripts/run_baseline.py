#!/usr/bin/env python
"""Run a baseline (K-Means versus BA) experiment.

Usage: python scripts/run_baseline.py [EXPERIMENT_ID] [--overwrite | --resume] [--verbose]

Without an ID this runs ``baseline_pilot``. Options are those of ``ba-clustering experiment``.
"""

from __future__ import annotations

import sys

from ba_clustering.cli import main

DEFAULT_EXPERIMENT = "baseline_pilot"

if __name__ == "__main__":
    arguments = sys.argv[1:]
    if not arguments or arguments[0].startswith("-"):
        arguments = [DEFAULT_EXPERIMENT, *arguments]
    sys.exit(main(["experiment", *arguments]))
