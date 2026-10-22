"""Optional CUDA stage annotations used by the Spike Mamba kernels.

The released LRA package does not include the standalone profiling experiments,
so stage annotations default to a zero-overhead no-op context manager.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator


@contextmanager
def cuda_stage(_name: str) -> Iterator[None]:
    """Mark a model stage without changing execution."""
    yield
