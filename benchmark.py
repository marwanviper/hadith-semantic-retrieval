"""
Benchmark entry point.

Thin wrapper around `app.py benchmark` for the convenience workflow:
  python benchmark.py --algorithm brute_force
  python benchmark.py --algorithm ivf
  python benchmark.py --algorithm hnsw
  python benchmark.py --algorithm pq
  python benchmark.py --algorithm ivf_pq
  python benchmark.py --algorithm all

All arguments are forwarded to the main CLI's benchmark subcommand.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Inject 'benchmark' into argv so app.py's subcommand parser routes correctly
sys.argv = [sys.argv[0], "benchmark"] + sys.argv[1:]

# Re-use the main CLI
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app import main

if __name__ == "__main__":
    main()
