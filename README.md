# Spike Mamba for Long Range Arena

This repository contains only the Spike Mamba model and its Long Range Arena
(LRA) experiments. It supports the five standard LRA tasks: ListOps, Text,
Retrieval, Image, and Pathfinder.

## Installation

The CUDA extensions require a CUDA-enabled PyTorch installation and a working
CUDA compiler toolchain.

```bash
pip install -r requirements.txt
bash build_extensions.sh
python tests/test_import.py
```

## Data layout

Pass a common data root to the reproduction script. The expected layout is:

```text
DATA_ROOT/
├── cifar/                         # optional torchvision download/cache
├── imdb/                          # processed IMDb cache
├── listops-1000/                  # or lra_release/listops-1000/
└── lra_release/lra_release/
    ├── pathfinder32/
    └── tsv_data/                  # AAN retrieval data
```

The loaders also accept the compatible fallback paths documented by
`scripts/run_lra.sh`.

## LRA experiments

Canonical hyperparameters are listed in
[`configs/lra.yaml`](configs/lra.yaml). Run one task
with:

```bash
bash scripts/run_lra.sh listops "$DATA_ROOT" 0
bash scripts/run_lra.sh text "$DATA_ROOT" 0
bash scripts/run_lra.sh retrieval "$DATA_ROOT" 0
bash scripts/run_lra.sh image "$DATA_ROOT" 0
bash scripts/run_lra.sh pathfinder "$DATA_ROOT" 0
```

The optional fourth argument changes the output root, and the fifth overrides
the random seed:

```bash
bash scripts/run_lra.sh pathfinder "$DATA_ROOT" 0 outputs/lra_runs 42
```

For a custom LRA configuration, call the Python entry point directly:

```bash
python train_lra.py --task listops --data_dir /path/to/listops-1000
```

Check all options with `python train_lra.py --help`. Training artifacts are
written beneath `outputs/` by the scripts and are ignored by Git.

## Model implementation

The active path is:

```text
SpikeMambaNet
  -> spike_mamba/e_model.py::Mamba
  -> selective_scan_flat_ch_fn
  -> selective_scan_cuda.fwd_flat_ch / bwd_flat_ch
  -> restore_dense_cuda_nodecay
```

No pretrained checkpoints, generated logs, or experiment outputs are included.

## License

This code is released under the Apache License 2.0. See
[`LICENSE`](LICENSE). Third-party copyright notices in derived Mamba selective
scan sources are intentionally retained.
