#!/usr/bin/env python
"""Run a parameter-sweep experiment.

Usage: python scripts/run_parameter_sweep.py [EXPERIMENT_ID] [--overwrite | --resume] [--verbose]

Without an ID this runs ``ofat_pilot``. Options are those of ``ba-clustering experiment``.
"""

from __future__ import annotations

import sys

from ba_clustering.cli import main

DEFAULT_EXPERIMENT = "ofat_pilot"

if __name__ == "__main__":
    arguments = sys.argv[1:]
    if not arguments or arguments[0].startswith("-"):
        arguments = [DEFAULT_EXPERIMENT, *arguments]
    sys.exit(main(["experiment", *arguments]))
