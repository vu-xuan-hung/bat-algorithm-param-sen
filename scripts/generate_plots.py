#!/usr/bin/env python
"""Generate figures from a saved results directory.

Usage: python scripts/generate_plots.py results/<experiment_id>
"""

from __future__ import annotations

import sys

from ba_clustering.cli import main

if __name__ == "__main__":
    sys.exit(main(["plot", *sys.argv[1:]]))
