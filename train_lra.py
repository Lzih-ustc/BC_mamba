#!/usr/bin/env python3
"""LRA training entry point for Spike Mamba."""
from __future__ import annotations

import sys
from pathlib import Path


LRA_TASKS = {"listops", "text", "retrieval", "image", "pathfinder"}


def main() -> None:
    root = Path(__file__).resolve().parent
    # e_model.py and the custom autograd interface use historical top-level
    # imports; make this package directory their only local import root.
    sys.path.insert(0, str(root / "spike_mamba"))
    import lra_impl

    argv = sys.argv[1:]
    if "--task" in argv:
        i = argv.index("--task")
        if i + 1 == len(argv) or argv[i + 1] not in LRA_TASKS:
            raise SystemExit(f"--task must be one of: {', '.join(sorted(LRA_TASKS))}")

    sys.argv = [sys.argv[0], *argv]
    lra_impl.main()


if __name__ == "__main__":
    main()
