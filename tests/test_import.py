"""Quick import check; run after build_extensions.sh."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "spike_mamba"))

import e_model  # noqa: E402
import selective_scan_interface  # noqa: E402
import selective_scan_cuda  # noqa: E402
import extract_events_cuda  # noqa: E402
import restore_dense_cuda_nodecay  # noqa: E402

assert e_model.Mamba.__name__ == "Mamba"
assert hasattr(selective_scan_interface, "selective_scan_flat_ch_fn")
assert hasattr(restore_dense_cuda_nodecay, "restore_dense_fwd")
for extension in (selective_scan_cuda, extract_events_cuda, restore_dense_cuda_nodecay):
    assert Path(extension.__file__).resolve().parent == ROOT / "spike_mamba", (
        f"Expected repository-local extension, got {extension.__file__}"
    )
print("Spike-Mamba flat CUDA interface import: OK")
