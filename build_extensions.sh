#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# CUDA translation units are memory-heavy; callers may override this value.
export MAX_JOBS="${MAX_JOBS:-2}"
(cd "$ROOT/csrc/selective_scan" && python setup.py build_ext --inplace)
(cd "$ROOT/csrc/extract" && python setup.py build_ext --inplace)
(cd "$ROOT/csrc/r_nodecay" && python setup.py build_ext --inplace)
find "$ROOT/csrc/selective_scan" "$ROOT/csrc/extract" "$ROOT/csrc/r_nodecay" \
  -maxdepth 1 -name '*.so' -exec cp {} "$ROOT/spike_mamba/" \;
