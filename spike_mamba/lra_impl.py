#!/usr/bin/env python3
"""
AAAI LRA runner for the flat event-driven Spike-Mamba model.

Supported tasks: listops, text, retrieval, image, pathfinder.
"""
from __future__ import annotations

import argparse
import csv
import copy
import json
import math
import os
import pickle
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List, Sequence, Union

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from PIL import Image, ImageFile
from e_model import Mamba, SpikeSamplerSurrogate
# The AAAI LRA package retains flat Spike-Mamba only.  The historical
# SpikeMamba1Lite image-only import is deliberately omitted.

ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    limit = 2**31 - 1
    while limit > 0:
        try:
            csv.field_size_limit(limit)
            break
        except OverflowError:
            limit //= 10

try:
    from datasets import load_dataset as hf_load_dataset, load_from_disk as hf_load_from_disk
except Exception:
    hf_load_dataset = None
    hf_load_from_disk = None


TASK_CONFIGS: Dict[str, Dict[str, Any]] = {
    "image": {
        "input_type": "continuous",
        "num_classes": 10,
        "vocab_size": None,
        "max_length": 1024,
        "batch_size": 128,
        "epochs": 200,
        "lr": 1e-3,
        "d_model": 256,
        "n_layers": 6,
        "d_state": 64,
        "expand": 2,
        "pool": "mean",
    },
    "text": {
        "input_type": "token",
        "num_classes": 2,
        "vocab_size": 256,
        "max_length": 4096,
        "batch_size": 32,
        "epochs": 100,
        "lr": 5e-4,
        "d_model": 64,
        "n_layers": 6,
        "d_state": 64,
        "expand": 2,
        "pool": "mean",
    },
    "listops": {
        "input_type": "token",
        "num_classes": 10,
        "vocab_size": None,
        "max_length": 2048,
        "batch_size": 64,
        "epochs": 50,
        "lr": 1e-4,
        "d_model": 64,
        "n_layers": 10,
        "d_state": 64,
        "expand": 3,
        "pool": "mean",
    },
    "retrieval": {
        "input_type": "token",
        "num_classes": 2,
        "vocab_size": 256,
        "max_length": 4000,
        "batch_size": 16,
        "epochs": 20,
        "lr": 1e-3,
        "d_model": 256,
        "n_layers": 6,
        "d_state": 64,
        "expand": 2,
        "pool": "mean",
    },
    "pathfinder": {
        "input_type": "continuous",
        "num_classes": 2,
        "vocab_size": None,
        "max_length": 1024,
        "batch_size": 32,
        "epochs": 200,
        "lr": 1e-4,
        "d_model": 128,
        "n_layers": 6,
        "d_state": 64,
        "expand": 4,
        "pool": "mean",
    },
}

TOKEN_FIELD_CANDIDATES = [
    "input_ids", "inputs", "tokens", "input", "sequence", "src", "text", "review",
    "Source", "source", "input_text", "document",
]
LABEL_FIELD_CANDIDATES = [
    "label", "labels", "target", "targets", "answer", "class",
    "Target", "Label", "Class",
]
PAIR_TOKEN_FIELD_CANDIDATES = [
    ("input_ids_0", "input_ids_1"),
    ("inputs_0", "inputs_1"),
    ("token_ids_0", "token_ids_1"),
    ("input0", "input1"),
    ("inputs0", "inputs1"),
    ("input1", "input2"),
    ("inputs1", "inputs2"),
    ("text1", "text2"),
    ("text_a", "text_b"),
    ("document1", "document2"),
    ("doc1", "doc2"),
    ("paper1", "paper2"),
    ("paper1_text", "paper2_text"),
    ("source", "target"),
    ("left", "right"),
    ("premise", "hypothesis"),
    ("query", "document"),
]
PAIR_CONTAINER_FIELD_CANDIDATES = [
    "inputs", "input_ids", "pair", "documents", "texts", "sequence_pair", "input_pair",
]

_LISTOPS_TOKEN_RE = re.compile(r"\[|\]|-?\d+|[A-Za-z_]+")



def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def encode_bernoulli(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    return torch.bernoulli(x)


def encode_poisson(x: torch.Tensor, T: int) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    p = x.unsqueeze(1).expand(-1, T, -1)  # (B,T,L)
    return torch.bernoulli(p)


def apply_task_defaults(args):
    cfg = TASK_CONFIGS[args.task]
    if args.batch_size is None:
        args.batch_size = cfg["batch_size"]
    if args.epochs is None:
        args.epochs = cfg["epochs"]
    if args.lr is None:
        args.lr = cfg["lr"]
    if args.d_model is None:
        args.d_model = cfg["d_model"]
    if args.n_layers is None:
        args.n_layers = cfg["n_layers"]
    if args.d_state is None:
        args.d_state = cfg["d_state"]
    if args.expand is None:
        args.expand = cfg["expand"]
    if args.pool is None:
        args.pool = cfg["pool"]
    if args.input_type is None:
        args.input_type = cfg["input_type"]
    if args.max_length is None:
        args.max_length = cfg["max_length"]
    if args.num_classes is None:
        args.num_classes = cfg["num_classes"]
    if args.vocab_size is None:
        args.vocab_size = cfg["vocab_size"]
    return args



class CIFAR10SeqCollator:
    def __init__(self, input_mode: str, T: int):
        self.input_mode = input_mode
        self.T = T

    def __call__(self, batch):
        xs, ys = zip(*batch)
        x = torch.stack(xs, dim=0)
        y = torch.tensor(ys, dtype=torch.long)
        x = x.view(x.size(0), -1)
        if self.input_mode == "cont":
            return x, y
        elif self.input_mode == "bernoulli":
            return encode_bernoulli(x), y
        elif self.input_mode == "poisson":
            return encode_poisson(x, T=self.T), y
        raise ValueError(f"Unknown input_mode={self.input_mode}")


def build_cifar10_image_loaders(
    data_dir: str,
    batch_size: int,
    input_mode: str,
    T: int,
) -> Tuple[DataLoader, DataLoader]:
    tfm = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])
    train = datasets.CIFAR10(data_dir, train=True, download=True, transform=tfm)
    test = datasets.CIFAR10(data_dir, train=False, download=True, transform=tfm)

    collate = CIFAR10SeqCollator(input_mode=input_mode, T=T)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, collate_fn=collate)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate)
    return train_loader, test_loader


def build_s4_cifar10_image_loaders(
    data_dir: str,
    batch_size: int,
    val_split: float = 0.1,
    seed: int = 42,
    num_workers: int = 0,
    aug: str = "none",
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """CIFAR-10 image pipeline matched to s4-main experiment=lra/s4-cifar."""
    data_root = Path(data_dir).expanduser()
    cifar_root = data_root if data_root.name == "cifar" else data_root / "cifar"

    aug = str(aug or "none").lower()
    train_ops = []
    if aug in ("crop_flip", "crop_flip_erasing", "crop_flip_affine"):
        train_ops.extend([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])
    if aug == "crop_flip_affine":
        train_ops.append(
            transforms.RandomAffine(
                degrees=10,
                translate=(0.08, 0.08),
                scale=(0.95, 1.05),
                fill=0,
            )
        )
    elif aug not in ("none", "crop_flip", "crop_flip_erasing"):
        raise ValueError(
            f"Unknown --s4_image_aug={aug}. Expected one of: none, crop_flip, "
            "crop_flip_erasing, crop_flip_affine"
        )

    train_ops.extend([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=122.6 / 255.0, std=61.0 / 255.0),
    ])
    if aug == "crop_flip_erasing":
        train_ops.append(
            transforms.RandomErasing(
                p=0.25,
                scale=(0.02, 0.12),
                ratio=(0.3, 3.3),
                value=0.0,
            )
        )
    train_ops.append(transforms.Lambda(lambda x: x.view(1, 32 * 32).t()))
    transform_train = transforms.Compose(train_ops)
    transform_eval = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=122.6 / 255.0, std=61.0 / 255.0),
        transforms.Lambda(lambda x: x.view(1, 32 * 32).t()),
    ])

    train_full = datasets.CIFAR10(str(cifar_root), train=True, download=True, transform=transform_train)
    test = datasets.CIFAR10(str(cifar_root), train=False, download=True, transform=transform_eval)

    train_len = int(len(train_full) * (1.0 - val_split))
    val_len = len(train_full) - train_len
    train, val = torch.utils.data.random_split(
        train_full,
        (train_len, val_len),
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader


class HFDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.ds[int(idx)]


def _first_present(d: Dict[str, Any], candidates: Sequence[str]) -> Optional[str]:
    for k in candidates:
        if k in d:
            return k
    return None


def _encode_text_to_bytes(x: Any) -> List[int]:
    if isinstance(x, bytes):
        return list(x)
    if isinstance(x, str):
        s = x.strip()
        if (s.startswith("b'") and s.endswith("'")) or (s.startswith('b"') and s.endswith('"')):
            try:
                import ast
                literal = ast.literal_eval(s)
                if isinstance(literal, (bytes, bytearray)):
                    return list(bytes(literal))
            except Exception:
                pass
        return list(s.encode("utf-8", errors="ignore"))
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().reshape(-1).long().tolist()
    if isinstance(x, (list, tuple)):
        flat: List[int] = []
        for v in x:
            if isinstance(v, (list, tuple, torch.Tensor)):
                flat.extend(_encode_text_to_bytes(v))
            else:
                flat.append(int(v))
        return flat
    return [int(x)]


def _to_int_label(x: Any) -> int:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().reshape(-1)[0].item()
    if isinstance(x, bytes):
        x = x.decode("utf-8", errors="ignore")
    if isinstance(x, str):
        x = x.strip()
        if x == "":
            raise ValueError("Empty string label encountered")
        try:
            return int(x)
        except ValueError:
            return int(float(x))
    return int(x)


class LocalTSVDataset(torch.utils.data.Dataset):
    def __init__(self, path: Union[str, Path]):
        self.path = str(path)
        self.samples: List[Dict[str, Any]] = []

        def _looks_like_header(row: List[str]) -> bool:
            if not row:
                return False
            row_norm = [str(x).strip().lower() for x in row]
            strong_header_tokens = {
                "label", "labels", "target", "targets", "answer", "class",
                "source",
                "text1", "text2", "text_a", "text_b",
                "document1", "document2", "doc1", "doc2",
                "paper1", "paper2", "paper1_text", "paper2_text",
                "input1", "input2", "inputs1", "inputs2",
                "input_ids_0", "input_ids_1",
            }
            if any(tok in strong_header_tokens for tok in row_norm):
                return True
            first = row_norm[0]
            if first in {"0", "1", "0.0", "1.0"}:
                return False
            docid_pat = re.compile(r"^[a-z]\d{2}-\d{4}$", re.I)
            if any(docid_pat.match(x) for x in row_norm[:3]):
                return False
            if any(x.startswith("b'") or x.startswith('b"') for x in row_norm):
                return False
            return False

        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter="\t")
            rows = [row for row in reader if row and any(str(x).strip() != "" for x in row)]

        if not rows:
            raise RuntimeError(f"Parsed 0 rows from TSV: {path}")

        first_row = rows[0]
        has_header = _looks_like_header(first_row)

        if has_header:
            header = [str(x) for x in first_row]
            data_rows = rows[1:]
        else:
            ncols = len(first_row)
            if ncols >= 5:
                header = ["label", "paper1", "paper2", "text1", "text2"] + [f"extra_{i}" for i in range(5, ncols)]
            elif ncols == 4:
                header = ["label", "paper1", "text1", "text2"]
            elif ncols == 3:
                header = ["label", "text1", "text2"]
            else:
                header = [f"col_{i}" for i in range(ncols)]
            data_rows = rows

        for row in data_rows:
            if len(row) < len(header):
                row = list(row) + [""] * (len(header) - len(row))
            elif len(row) > len(header):
                row = list(row[:len(header) - 1]) + ["\t".join(row[len(header) - 1:])]
            self.samples.append(dict(zip(header, row)))

        if not self.samples:
            raise RuntimeError(f"Parsed 0 rows from TSV: {path}")
        print(f"[INFO] LocalTSVDataset loaded {len(self.samples)} rows from {path} | header_detected={has_header} | columns={header[:6]}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[int(idx)]


def _find_existing_file(root: Path, names: Sequence[str]) -> Optional[Path]:
    for name in names:
        p = root / name
        if p.exists() and p.is_file():
            return p
    return None


def _resolve_search_root_for_local_tsv(root: Path) -> Path:
    if root.is_dir() and (root / "tsv_data").is_dir():
        return root / "tsv_data"
    return root


def _find_split_file(root: Path, split: str, fallback_exact: Sequence[str] = ()) -> Optional[Path]:
    split = str(split).lower()
    exact_names = [
        f"{split}.tsv",
        f"basic_{split}.tsv",
        f"new_aan_pairs.{split}.tsv",
        *fallback_exact,
    ]
    p = _find_existing_file(root, exact_names)
    if p is not None:
        return p

    candidates = []
    for q in root.glob("*.tsv"):
        name = q.name.lower()
        if split in name:
            candidates.append(q)

    if not candidates:
        return None

    filtered = [q for q in candidates if "sample" not in q.name.lower() and "debug" not in q.name.lower()]
    if filtered:
        candidates = filtered

    def score(path: Path):
        name = path.name.lower()
        return (
            0 if name == f"new_aan_pairs.{split}.tsv" else 1,
            0 if name == f"{split}.tsv" else 1,
            0 if name == f"basic_{split}.tsv" else 1,
            0 if f".{split}." in name else 1,
            len(name),
            name,
        )

    return sorted(candidates, key=score)[0]


def _resolve_local_tsv_splits(
    hf_local_path: Union[str, Path],
    hf_train_split: str,
    hf_test_split: str,
):
    root = Path(hf_local_path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"hf_local_path does not exist: {root}")

    root = _resolve_search_root_for_local_tsv(root)

    train_path = _find_split_file(
        root,
        hf_train_split,
        fallback_exact=("basic_train.tsv", "train.tsv", "new_aan_pairs.train.tsv"),
    )
    test_path = _find_split_file(
        root,
        hf_test_split,
        fallback_exact=("basic_test.tsv", "test.tsv", f"new_aan_pairs.{hf_test_split}.tsv"),
    )

    if test_path is None and str(hf_test_split).lower() == "test":
        test_path = (
            _find_split_file(root, "eval", fallback_exact=("basic_eval.tsv", "eval.tsv", "new_aan_pairs.eval.tsv"))
            or _find_split_file(root, "val", fallback_exact=("basic_val.tsv", "val.tsv", "validation.tsv"))
            or _find_split_file(root, "validation", fallback_exact=("validation.tsv",))
        )

    if train_path is None or test_path is None:
        present = sorted(p.name for p in root.iterdir() if p.is_file())
        raise FileNotFoundError(
            "Could not resolve local TSV splits under "
            f"{root}. Need train/test TSV files. Present files: {present}"
        )

    print(f"[INFO] Local TSV train split: {train_path}")
    print(f"[INFO] Local TSV eval/test split: {test_path}")
    return LocalTSVDataset(train_path), LocalTSVDataset(test_path)


def _load_hf_splits(
    data_dir: str,
    hf_dataset_name: str,
    hf_config_name: Optional[str],
    hf_train_split: str,
    hf_test_split: str,
    hf_local_path: Optional[str] = None,
):
    ds_kwargs = {}
    if data_dir:
        ds_kwargs["cache_dir"] = data_dir

    if hf_local_path:
        root = Path(hf_local_path).expanduser().resolve()

        tsv_root = _resolve_search_root_for_local_tsv(root)
        if tsv_root.is_dir() and any(tsv_root.glob("*.tsv")):
            return _resolve_local_tsv_splits(tsv_root, hf_train_split=hf_train_split, hf_test_split=hf_test_split)

        if hf_load_from_disk is None:
            raise ImportError("Please install the Hugging Face datasets package: pip install datasets")
        ds = hf_load_from_disk(str(root))
        if isinstance(ds, dict):
            train_ds = ds[hf_train_split]
            test_ds = ds[hf_test_split]
        else:
            try:
                train_ds = ds[hf_train_split]
                test_ds = ds[hf_test_split]
            except Exception as e:
                raise ValueError(
                    f"hf_local_path={hf_local_path} is neither a TSV split directory nor a DatasetDict containing "
                    f"splits '{hf_train_split}'/'{hf_test_split}'"
                ) from e
        return train_ds, test_ds

    if hf_load_dataset is None:
        raise ImportError("Please install the Hugging Face datasets package: pip install datasets")

    if hf_config_name is None or str(hf_config_name).lower() in ("", "none", "null"):
        train_ds = hf_load_dataset(hf_dataset_name, split=hf_train_split, **ds_kwargs)
        test_ds = hf_load_dataset(hf_dataset_name, split=hf_test_split, **ds_kwargs)
    else:
        train_ds = hf_load_dataset(hf_dataset_name, hf_config_name, split=hf_train_split, **ds_kwargs)
        test_ds = hf_load_dataset(hf_dataset_name, hf_config_name, split=hf_test_split, **ds_kwargs)
    return train_ds, test_ds


def _listops_tokenize(x: Any) -> List[str]:
    if isinstance(x, bytes):
        x = x.decode("utf-8", errors="ignore")
    if isinstance(x, str):
        # Match the LRA reference preprocessing exactly: parentheses are
        # discarded and the closing bracket token is named "X".
        return x.translate({ord("]"): ord("X"), ord("("): None, ord(")"): None}).split()
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().reshape(-1).tolist()
    if isinstance(x, (list, tuple)):
        out: List[str] = []
        for v in x:
            out.extend(_listops_tokenize(v))
        return out
    return [str(x)]


def _build_listops_vocab(train_ds: Sequence[Any], token_field: str) -> Dict[str, int]:
    """Build the SSM-benchmark ListOps vocabulary from train data only."""
    vocab: Dict[str, int] = {"<pad>": 0, "<unk>": 1}
    vocab["<eos>"] = len(vocab)
    counts = Counter()
    for item in train_ds:
        counts.update(_listops_tokenize(item[token_field]))
    for tok, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        vocab[tok] = len(vocab)
    return vocab


def _detect_label_field(example: Dict[str, Any]) -> str:
    key = _first_present(example, LABEL_FIELD_CANDIDATES)
    if key is None:
        raise KeyError(f"Cannot detect label field from keys: {list(example.keys())}")
    return key


def _detect_token_field(example: Dict[str, Any]) -> str:
    key = _first_present(example, TOKEN_FIELD_CANDIDATES)
    if key is None:
        raise KeyError(f"Cannot detect token/text field from keys: {list(example.keys())}")
    return key


def _detect_pair_token_fields(example: Dict[str, Any]):
    for left_key, right_key in PAIR_TOKEN_FIELD_CANDIDATES:
        if left_key in example and right_key in example:
            return ("fields", left_key, right_key)
    for key in PAIR_CONTAINER_FIELD_CANDIDATES:
        if key not in example:
            continue
        value = example[key]
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return ("container", key, None)
    for key in example.keys():
        if key.endswith("_0"):
            peer = key[:-2] + "_1"
            if peer in example:
                return ("fields", key, peer)
        if key.endswith("1") and (key[:-1] + "2") in example:
            return ("fields", key, key[:-1] + "2")
    raise KeyError(f"Cannot detect pair token/text fields from keys: {list(example.keys())}")


def _extract_pair_tokens(item: Dict[str, Any], pair_spec):
    mode, key_a, key_b = pair_spec
    if mode == "fields":
        return item[key_a], item[key_b]
    pair_value = item[key_a]
    if not isinstance(pair_value, (list, tuple)) or len(pair_value) != 2:
        raise ValueError(f"Expected a length-2 pair in field '{key_a}', got {type(pair_value)}")
    return pair_value[0], pair_value[1]


class TextTokenCollator:
    def __init__(self, max_length: int, pad_id: int, label_field: str, token_field: str):
        self.max_length = max_length
        self.pad_id = pad_id
        self.label_field = label_field
        self.token_field = token_field

    def __call__(self, batch: Sequence[Dict[str, Any]]):
        xs, ys = [], []
        for item in batch:
            x = torch.tensor(_encode_text_to_bytes(item[self.token_field]), dtype=torch.long)
            x = pad_or_truncate_1d(x, self.max_length, self.pad_id)
            y = _to_int_label(item[self.label_field])
            xs.append(x)
            ys.append(y)
        return torch.stack(xs, dim=0), torch.tensor(ys, dtype=torch.long)


class SSMBenchmarkIMDBCollator:
    """Collate the cached character-level IMDb tokens exactly as ssm-benchmark.

    The reference pipeline appends EOS during preprocessing and pads each batch
    only to its longest sample.  The cached DatasetDict contains ``input_ids``
    and ``label`` already, so no byte re-encoding or new vocabulary is used.
    """

    def __init__(self, pad_id: int):
        self.pad_id = int(pad_id)

    def __call__(self, batch: Sequence[Dict[str, Any]]):
        xs = [torch.as_tensor(item["input_ids"], dtype=torch.long) for item in batch]
        ys = torch.tensor([_to_int_label(item["label"]) for item in batch], dtype=torch.long)
        return nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=self.pad_id), ys


class _SSMBenchmarkVocabCompat:
    """Minimal stand-in for the cached ``dataloaders.lra.SimpleVocab`` object."""

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __getitem__(self, token: str) -> int:
        return int(self.stoi[token])

    def __len__(self) -> int:
        return len(self.itos)


class _SSMBenchmarkVocabUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module == "dataloaders.lra" and name == "SimpleVocab":
            return _SSMBenchmarkVocabCompat
        return super().find_class(module, name)


def build_ssm_benchmark_imdb_loaders(
    data_dir: str,
    batch_size: int,
    cache_dir: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, int, int]:
    """Load the processed IMDb cache made by ``ssm-benchmark/dataloaders/lra.py``."""
    if hf_load_from_disk is None:
        raise ImportError("The ssm-benchmark IMDb compatibility pipeline requires the datasets package.")

    root = Path(data_dir).expanduser().resolve()
    cache = (
        Path(cache_dir).expanduser().resolve()
        if cache_dir
        else root / "cache" / "l_max-4096-level-char-min_freq-15-append_bos-False-append_eos-True"
    )
    if not (cache / "dataset_dict.json").is_file():
        raise FileNotFoundError(
            "Could not find the processed ssm-benchmark IMDb cache at "
            f"{cache}. Run the original IMDb setup once or pass --ssm_imdb_cache_dir."
        )

    dataset = hf_load_from_disk(str(cache))
    if "train" not in dataset or "test" not in dataset:
        raise ValueError(f"Expected train/test splits in {cache}, found {list(dataset.keys())}")

    vocab_path = cache / "vocab.pkl"
    if not vocab_path.is_file():
        raise FileNotFoundError(f"Missing reference vocabulary: {vocab_path}")
    with open(vocab_path, "rb") as f:
        vocab = _SSMBenchmarkVocabUnpickler(f).load()
    try:
        pad_id = int(vocab["<pad>"])
        vocab_size = len(vocab)
    except Exception as exc:
        raise ValueError(f"Could not read <pad> and vocabulary size from {vocab_path}") from exc

    collate = SSMBenchmarkIMDBCollator(pad_id=pad_id)
    train_loader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True, collate_fn=collate)
    test_loader = DataLoader(dataset["test"], batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=True, collate_fn=collate)
    print(
        f"[INFO] ssm-benchmark IMDb cache | train={len(dataset['train'])} | "
        f"test={len(dataset['test'])} | vocab={vocab_size} | pad_id={pad_id} | cache={cache}"
    )
    return train_loader, test_loader, 2, vocab_size


class SSMBenchmarkListOpsCollator:
    """Reference ListOps cache collator: dynamically pad cached token IDs."""

    def __init__(self, pad_id: int):
        self.pad_id = int(pad_id)

    def __call__(self, batch: Sequence[Dict[str, Any]]):
        xs = [torch.as_tensor(item["input_ids"], dtype=torch.long) for item in batch]
        ys = torch.tensor([_to_int_label(item["Target"]) for item in batch], dtype=torch.long)
        return nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=self.pad_id), ys


def build_ssm_benchmark_listops_loaders(
    data_dir: str,
    batch_size: int,
    cache_dir: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, int, int]:
    """Load the exact cached ListOps preprocessing made by ssm-benchmark."""
    if hf_load_from_disk is None:
        raise ImportError("The ssm-benchmark ListOps compatibility pipeline requires the datasets package.")

    root = Path(data_dir).expanduser().resolve()
    cache_name = "l_max-2048-append_bos-False-append_eos-True"
    cache = Path(cache_dir).expanduser().resolve() if cache_dir else root / cache_name / cache_name
    if not (cache / "dataset_dict.json").is_file():
        raise FileNotFoundError(
            "Could not find the processed ssm-benchmark ListOps cache at "
            f"{cache}. Run the original ListOps setup once or pass --ssm_listops_cache_dir."
        )

    dataset = hf_load_from_disk(str(cache))
    if "train" not in dataset or "test" not in dataset:
        raise ValueError(f"Expected train/test splits in {cache}, found {list(dataset.keys())}")
    with open(cache / "vocab.pkl", "rb") as f:
        vocab = _SSMBenchmarkVocabUnpickler(f).load()
    pad_id = int(vocab["<pad>"])
    vocab_size = len(vocab)

    collate = SSMBenchmarkListOpsCollator(pad_id=pad_id)
    train_loader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True, collate_fn=collate)
    test_loader = DataLoader(dataset["test"], batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=True, collate_fn=collate)
    print(
        f"[INFO] ssm-benchmark ListOps cache | train={len(dataset['train'])} | "
        f"test={len(dataset['test'])} | vocab={vocab_size} | pad_id={pad_id} | cache={cache}"
    )
    return train_loader, test_loader, 10, vocab_size


def build_text_loaders(
    data_dir: str,
    batch_size: int,
    max_length: int,
    hf_dataset_name: str,
    hf_config_name: Optional[str] = None,
    hf_train_split: str = "train",
    hf_test_split: str = "test",
    pad_id: int = 0,
    hf_local_path: Optional[str] = None
) -> Tuple[DataLoader, DataLoader, int, int]:
    train_ds, test_ds = _load_hf_splits(
        data_dir=data_dir,
        hf_dataset_name=hf_dataset_name,
        hf_config_name=hf_config_name,
        hf_train_split=hf_train_split,
        hf_test_split=hf_test_split,
        hf_local_path=hf_local_path,
    )

    train_wrap = HFDatasetWrapper(train_ds)
    test_wrap = HFDatasetWrapper(test_ds)
    example = train_wrap[0]
    label_field = _detect_label_field(example)
    token_field = _detect_token_field(example)

    collate = TextTokenCollator(
        max_length=max_length,
        pad_id=pad_id,
        label_field=label_field,
        token_field=token_field,
    )

    train_loader = DataLoader(train_wrap, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, collate_fn=collate)
    test_loader = DataLoader(test_wrap, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate)

    num_classes = len({_to_int_label(train_wrap[i][label_field]) for i in range(min(len(train_wrap), 2048))})
    vocab_size = 256
    return train_loader, test_loader, num_classes, vocab_size


class ListOpsCollator:
    def __init__(self, max_length: int, pad_id: int, eos_id: int, label_field: str, token_field: str, vocab: Dict[str, int]):
        self.max_length = max_length
        self.pad_id = pad_id
        self.eos_id = eos_id
        self.label_field = label_field
        self.token_field = token_field
        self.vocab = vocab
        self.unk_id = vocab.get("<unk>", 1)

    def __call__(self, batch: Sequence[Dict[str, Any]]):
        xs, ys = [], []
        for item in batch:
            toks = _listops_tokenize(item[self.token_field])
            # SSM-benchmark reserves one token for EOS, then its collate
            # function pads only to the longest sequence in this batch.
            ids = [self.vocab.get(tok, self.unk_id) for tok in toks[: max(0, self.max_length - 1)]]
            ids.append(self.eos_id)
            x = torch.tensor(ids, dtype=torch.long)
            y = _to_int_label(item[self.label_field])
            xs.append(x)
            ys.append(y)
        return (
            nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=self.pad_id),
            torch.tensor(ys, dtype=torch.long),
        )


def build_listops_loaders(
    data_dir: str,
    batch_size: int,
    max_length: int,
    hf_dataset_name: str,
    hf_config_name: Optional[str] = None,
    hf_train_split: str = "train",
    hf_test_split: str = "test",
    pad_id: int = 0,
    hf_local_path: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, int, int]:
    train_ds, test_ds = _load_hf_splits(
        data_dir=data_dir,
        hf_dataset_name=hf_dataset_name,
        hf_config_name=hf_config_name,
        hf_train_split=hf_train_split,
        hf_test_split=hf_test_split,
        hf_local_path=hf_local_path,
    )

    train_wrap = HFDatasetWrapper(train_ds)
    test_wrap = HFDatasetWrapper(test_ds)
    example = train_wrap[0]
    label_field = _detect_label_field(example)
    token_field = _detect_token_field(example)

    vocab = _build_listops_vocab(train_ds, token_field=token_field)
    print(f"[INFO] SSM-benchmark ListOps vocabulary: {len(vocab)} tokens (train-only, EOS enabled)")
    collate = ListOpsCollator(
        max_length=max_length,
        pad_id=vocab["<pad>"],
        eos_id=vocab["<eos>"],
        label_field=label_field,
        token_field=token_field,
        vocab=vocab,
    )

    train_loader = DataLoader(train_wrap, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, collate_fn=collate)
    test_loader = DataLoader(test_wrap, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate)

    labels = {_to_int_label(train_wrap[i][label_field]) for i in range(min(len(train_wrap), 8192))}
    num_classes = max(labels) + 1 if labels else 10
    vocab_size = len(vocab)
    return train_loader, test_loader, num_classes, vocab_size


def _retrieval_text_chars(value: Any) -> List[str]:
    """Match S4-main AAN tokenization: one Unicode character per token."""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    return list(str(value))


def _build_retrieval_vocab(train_ds: Sequence[Any], pair_spec) -> Dict[str, int]:
    """Build the AAN character vocabulary from training documents only."""
    counts = Counter()
    for item in train_ds:
        left, right = _extract_pair_tokens(item, pair_spec)
        counts.update(_retrieval_text_chars(left))
        counts.update(_retrieval_text_chars(right))

    vocab = {"<pad>": 0, "<unk>": 1, "<eos>": 2}
    for token, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        vocab[token] = len(vocab)
    return vocab


class RetrievalPairCollator:
    def __init__(self, max_length: int, pad_id: int, eos_id: int, unk_id: int, label_field: str, pair_spec, vocab: Optional[Dict[str, int]], pretokenized: bool = False):
        self.max_length = max_length
        self.pad_id = pad_id
        self.eos_id = eos_id
        self.unk_id = unk_id
        self.label_field = label_field
        self.pair_spec = pair_spec
        self.vocab = vocab
        self.pretokenized = bool(pretokenized)

    def _encode(self, value: Any) -> torch.Tensor:
        if self.pretokenized:
            if isinstance(value, torch.Tensor):
                return value.detach().clone().long()[: self.max_length]
            return torch.tensor(value, dtype=torch.long)[: self.max_length]
        # S4-main reserves one position for EOS then pads only to this batch's
        # maximum sequence length.
        chars = _retrieval_text_chars(value)[: max(0, self.max_length - 1)]
        ids = [self.vocab.get(ch, self.unk_id) for ch in chars]
        ids.append(self.eos_id)
        return torch.tensor(ids, dtype=torch.long)

    def __call__(self, batch: Sequence[Dict[str, Any]]):
        xs1, xs2, ys = [], [], []
        for item in batch:
            left_raw, right_raw = _extract_pair_tokens(item, self.pair_spec)
            x1 = self._encode(left_raw)
            x2 = self._encode(right_raw)
            y = _to_int_label(item[self.label_field])
            xs1.append(x1)
            xs2.append(x2)
            ys.append(y)
        lengths1 = torch.tensor([len(x) for x in xs1], dtype=torch.long)
        lengths2 = torch.tensor([len(x) for x in xs2], dtype=torch.long)
        x1 = nn.utils.rnn.pad_sequence(xs1, batch_first=True, padding_value=self.pad_id)
        x2 = nn.utils.rnn.pad_sequence(xs2, batch_first=True, padding_value=self.pad_id)
        # A shared length lets both documents be encoded in the same shape,
        # exactly as S4-main's AAN collator does.
        length = max(x1.size(1), x2.size(1))
        x1 = F.pad(x1, (0, length - x1.size(1)), value=self.pad_id)
        x2 = F.pad(x2, (0, length - x2.size(1)), value=self.pad_id)
        return (x1, x2, lengths1, lengths2), torch.tensor(ys, dtype=torch.long)


def build_retrieval_loaders(
    data_dir: str,
    batch_size: int,
    max_length: int,
    hf_dataset_name: str,
    hf_config_name: Optional[str] = None,
    hf_train_split: str = "train",
    hf_test_split: str = "test",
    pad_id: int = 0,
    hf_local_path: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, int, int]:
    # Reuse S4-main's finalized AAN cache when it is available. It contains
    # the canonical train-only 98-character vocabulary, EOS, and token IDs;
    # this also avoids reparsing the 8 GB raw training TSV per experiment.
    s4_cache = None
    if hf_local_path:
        candidate = Path(hf_local_path).expanduser().resolve() / (
            f"l_max-{max_length}-append_bos-False-append_eos-True"
        )
        if candidate.is_dir():
            s4_cache = candidate

    if s4_cache is not None:
        if hf_load_from_disk is None:
            raise ImportError("Please install the Hugging Face datasets package: pip install datasets")
        dataset = hf_load_from_disk(str(s4_cache))
        train_ds, test_ds = dataset["train"], dataset[hf_test_split]
        pretokenized = True
        vocab = None
        vocab_size = 98
        pad_id, eos_id, unk_id = 0, 2, 1
        print(f"[INFO] Reusing S4-main AAN cache: {s4_cache} | vocab_size={vocab_size}")
    else:
        train_ds, test_ds = _load_hf_splits(
            data_dir=data_dir,
            hf_dataset_name=hf_dataset_name,
            hf_config_name=hf_config_name,
            hf_train_split=hf_train_split,
            hf_test_split=hf_test_split,
            hf_local_path=hf_local_path,
        )
        pretokenized = False

    train_wrap = HFDatasetWrapper(train_ds)
    test_wrap = HFDatasetWrapper(test_ds)
    example = train_wrap[0]
    label_field = _detect_label_field(example)
    pair_spec = _detect_pair_token_fields(example)

    if not pretokenized:
        vocab = _build_retrieval_vocab(train_wrap, pair_spec)
        print(f"[INFO] S4-style AAN character vocabulary: {len(vocab)} tokens")
        vocab_size = len(vocab)
        pad_id, eos_id, unk_id = vocab["<pad>"], vocab["<eos>"], vocab["<unk>"]
    collate = RetrievalPairCollator(
        max_length=max_length,
        pad_id=pad_id,
        eos_id=eos_id,
        unk_id=unk_id,
        label_field=label_field,
        pair_spec=pair_spec,
        vocab=vocab,
        pretokenized=pretokenized,
    )

    train_loader = DataLoader(train_wrap, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, collate_fn=collate)
    test_loader = DataLoader(test_wrap, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate)

    labels = {_to_int_label(train_wrap[i][label_field]) for i in range(min(len(train_wrap), 8192))}
    num_classes = max(labels) + 1 if labels else 2
    return train_loader, test_loader, num_classes, vocab_size


class PathfinderSeqCollator:
    def __init__(self, input_mode: str, T: int, center: bool = False):
        self.input_mode = input_mode
        self.T = T
        self.center = bool(center)

    def __call__(self, batch):
        xs, ys = zip(*batch)
        x = torch.stack(xs, dim=0)
        if self.center:
            x = x.sub(0.5).div(0.5)
        y = torch.tensor(ys, dtype=torch.long)
        if self.input_mode == "cont":
            return x, y
        elif self.input_mode == "bernoulli":
            return encode_bernoulli(x), y
        elif self.input_mode == "poisson":
            return encode_poisson(x, T=self.T), y
        raise ValueError(f"Unknown input_mode={self.input_mode}")


def quantize_pathfinder_to_tokens(x: torch.Tensor, num_bins: int) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    if num_bins < 2:
        raise ValueError(f"pathfinder num_bins must be >= 2, got {num_bins}")
    ids = torch.round(x * float(num_bins - 1)).to(torch.long) + 1
    return ids.clamp_(1, num_bins)


class PathfinderTokenCollator:
    def __init__(self, num_bins: int, pad_id: int = 0):
        self.num_bins = int(num_bins)
        self.pad_id = int(pad_id)
        if self.pad_id != 0:
            raise ValueError("Pathfinder tokenization reserves pad_id=0; please keep --pad_id 0")

    def __call__(self, batch):
        xs, ys = zip(*batch)
        x = torch.stack(xs, dim=0)
        y = torch.tensor(ys, dtype=torch.long)
        x = quantize_pathfinder_to_tokens(x, num_bins=self.num_bins)
        return x, y


class LocalPathfinderDataset(torch.utils.data.Dataset):
    LEVEL_TO_SUBDIR = {
        "easy": "curv_baseline",
        "intermediate": "curv_contour_length_9",
        "hard": "curv_contour_length_14",
    }
    MANIFEST_VERSION = 2

    def __init__(
        self,
        root: Union[str, Path],
        resolution: int = 32,
        level: str = "hard",
        split: str = "train",
        split_seed: int = 123,
        split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    ):
        self.root = Path(root).expanduser().resolve()
        self.resolution = int(resolution)
        self.level = str(level)
        self.split = str(split)
        self.split_seed = int(split_seed)
        self.split_ratios = split_ratios

        self.data_dir = self._resolve_data_dir(self.root, self.resolution, self.level)
        self.samples = self._load_samples_with_cached_split()
        if not self.samples:
            raise RuntimeError(f"No Pathfinder samples found for split={self.split} under {self.data_dir}")
        print(
            f"[INFO] LocalPathfinderDataset split={self.split} res={self.resolution} "
            f"level={self.level} samples={len(self.samples)} data_dir={self.data_dir}"
        )

    @classmethod
    def _resolve_data_dir(cls, root: Path, resolution: int, level: str) -> Path:
        level_dir = cls.LEVEL_TO_SUBDIR.get(level, level)
        candidates = []

        if (root / "metadata").is_dir():
            candidates.append(root)

        pf_dir = root / f"pathfinder{resolution}"
        if pf_dir.is_dir():
            if level == "all":
                return pf_dir.resolve()
            candidates.append(pf_dir / level_dir)
            candidates.append(pf_dir)

        candidates.append(root / level_dir)

        for child in root.glob(f"**/pathfinder{resolution}"):
            candidates.append(child / level_dir)
            candidates.append(child)

        for cand in candidates:
            if cand.is_dir() and (cand / "metadata").is_dir():
                return cand.resolve()

        raise FileNotFoundError(
            f"Could not resolve Pathfinder data dir from root={root}, "
            f"resolution={resolution}, level={level}. Expected metadata/ under "
            f"pathfinder{resolution}/{level_dir} or a directly provided subset dir."
        )

    @staticmethod
    def _read_meta_lines(meta_path: Path) -> List[str]:
        try:
            txt = meta_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            txt = meta_path.read_bytes().decode("utf-8", errors="ignore")
        return [line.strip() for line in txt.splitlines() if line.strip()]

    @staticmethod
    def _is_valid_image_file(image_path: Path) -> bool:
        try:
            if not image_path.exists() or image_path.stat().st_size == 0:
                return False
            with Image.open(image_path) as img:
                img.verify()
            return True
        except Exception:
            return False

    def _build_manifest(self) -> List[Tuple[str, int]]:
        if self.level == "all":
            meta_files = sorted(self.data_dir.glob("*/metadata/*.npy"))
        else:
            meta_files = sorted((self.data_dir / "metadata").glob("*.npy"))
        if not meta_files:
            raise FileNotFoundError(f"No metadata/*.npy files found under {self.data_dir / 'metadata'}")

        samples: List[Tuple[str, int]] = []
        missing = 0
        invalid = 0
        for meta_path in meta_files:
            sample_root = meta_path.parent.parent
            for line in self._read_meta_lines(meta_path):
                parts = line.split(" ")
                if len(parts) < 4:
                    continue
                image_path = (sample_root / parts[0] / parts[1]).resolve()
                if not image_path.exists():
                    missing += 1
                    continue
                if not self._is_valid_image_file(image_path):
                    invalid += 1
                    continue
                label = int(float(parts[3]))
                samples.append((str(image_path), label))

        if not samples:
            raise RuntimeError(f"Parsed 0 valid Pathfinder samples from {self.data_dir}")
        if missing:
            print(f"[WARN] Skipped {missing} Pathfinder entries with missing image files under {self.data_dir}")
        if invalid:
            print(f"[WARN] Skipped {invalid} Pathfinder entries with unreadable/corrupt image files under {self.data_dir}")
        return samples

    def _manifest_cache_path(self) -> Path:
        return self.data_dir / f".pathfinder_manifest_v{self.MANIFEST_VERSION}_res{self.resolution}_{self.level}.pt"

    def _split_cache_path(self) -> Path:
        return self.data_dir / f".pathfinder_split_v{self.MANIFEST_VERSION}_res{self.resolution}_{self.level}_seed{self.split_seed}.json"

    def _load_samples_with_cached_split(self) -> List[Tuple[str, int]]:
        manifest_cache = self._manifest_cache_path()
        split_cache = self._split_cache_path()

        if manifest_cache.exists():
            manifest = torch.load(manifest_cache)
        else:
            manifest = self._build_manifest()
            torch.save(manifest, manifest_cache)

        if split_cache.exists():
            split_indices = json.loads(split_cache.read_text(encoding="utf-8"))
        else:
            indices = list(range(len(manifest)))
            rnd = random.Random(self.split_seed)
            rnd.shuffle(indices)

            n = len(indices)
            n_train = int(n * self.split_ratios[0])
            n_val = int(n * self.split_ratios[1])
            n_test = n - n_train - n_val
            split_indices = {
                "train": indices[:n_train],
                "val": indices[n_train:n_train + n_val],
                "test": indices[n_train + n_val:n_train + n_val + n_test],
            }
            split_cache.write_text(json.dumps(split_indices), encoding="utf-8")

        key = self.split
        if key == "eval":
            key = "val"
        if key not in split_indices:
            raise KeyError(f"Unknown Pathfinder split '{self.split}'. Available: {list(split_indices.keys())}")
        return [manifest[i] for i in split_indices[key]]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        max_tries = min(16, len(self.samples))
        cur_idx = int(idx)
        last_err = None
        for _ in range(max_tries):
            image_path, label = self.samples[cur_idx]
            try:
                img = Image.open(image_path).convert("L")
                x = transforms.ToTensor()(img).view(-1)
                return x, int(label)
            except Exception as e:
                last_err = e
                cur_idx = (cur_idx + 1) % len(self.samples)
        raise RuntimeError(f"Failed to load a valid Pathfinder image after {max_tries} attempts. Last error: {last_err}")


def build_pathfinder_loaders(
    data_dir: str,
    batch_size: int,
    input_mode: str,
    T: int,
    pathfinder_resolution: int,
    pathfinder_level: str,
    input_type: str = "token",
    pathfinder_bins: int = 256,
    pad_id: int = 0,
    hf_local_path: Optional[str] = None,
    pathfinder_split_seed: int = 123,
    pathfinder_center: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    root = hf_local_path if hf_local_path is not None else data_dir
    train_ds = LocalPathfinderDataset(
        root=root,
        resolution=pathfinder_resolution,
        level=pathfinder_level,
        split="train",
        split_seed=pathfinder_split_seed,
    )
    test_ds = LocalPathfinderDataset(
        root=root,
        resolution=pathfinder_resolution,
        level=pathfinder_level,
        split="test",
        split_seed=pathfinder_split_seed,
    )

    if input_type == "token":
        collate = PathfinderTokenCollator(num_bins=pathfinder_bins, pad_id=pad_id)
    else:
        collate = PathfinderSeqCollator(input_mode=input_mode, T=T, center=pathfinder_center)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate)
    return train_loader, test_loader


def get_norm(d_model: int) -> nn.Module:
    if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        return nn.LayerNorm(d_model)

    try:
        from model import RMSNorm
        if RMSNorm is not None:
            return RMSNorm(d_model)
    except Exception:
        pass
    return nn.LayerNorm(d_model)


class ResidualPreNorm(nn.Module):
    def __init__(self, d_model: int, core: nn.Module, leak_alpha: float = 0.95):
        super().__init__()
        self.norm = get_norm(d_model)
        self.core = core
        assert hasattr(core, "to_spike"), "core must provide to_spike()"
        self.leak_alpha = leak_alpha

        self.last_rate = None
        self.last_maxchan = None
        self.last_sgc_loss = None

    def forward(self, s_in: torch.Tensor, v_in: torch.Tensor):
        dv = self.core(s_in)

        v_out = self.leak_alpha * v_in + dv

        s_out = self.core.to_spike(self.norm(v_out))
        self.last_rate = s_out.detach().float().mean()

        self.last_sgc_loss = None
        sgc_terms = []

        out_f = getattr(self.core, "last_out_f", None)
        if out_f is not None:
            sgc_terms.append(
                F.mse_loss(
                    F.softmax(dv.float(), dim=-1),
                    F.softmax(out_f.float(), dim=-1),
                )
            )

        z_main = getattr(self.core, "last_zxbcdt", None)
        z_smooth = getattr(self.core, "last_zxbcdt_f", None)
        if z_main is not None and z_smooth is not None:
            sgc_terms.append(
                F.mse_loss(
                    F.softmax(z_main.float(), dim=-1),
                    F.softmax(z_smooth.float(), dim=-1),
                )
            )

        if len(sgc_terms) > 0:
            self.last_sgc_loss = sum(sgc_terms) / len(sgc_terms)

        return s_out, v_out


class SequenceClassifierBase(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layers: int,
        d_state: int,
        d_conv: int,
        expand: int,
        spike_mode: str,
        prob_activation: str,
        pool: str,
        target_rate: float,
        routing_threshold: float,
        fixed_input_rate: bool,
        input_rate: float,
        num_classes: int,
        input_type: str,
        vocab_size: Optional[int],
        core_cls,
        input_dim: int = 1,
        head_input: str = "spike",
    ):
        super().__init__()
        import math

        assert pool in ("last", "mean", "attn")
        assert input_type in ("continuous", "token")
        assert head_input in ("spike", "cont", "both")
        if not 0.0 < routing_threshold < 1.0:
            raise ValueError(f"routing_threshold must lie in (0, 1), got {routing_threshold}")
        if not 0.0 < target_rate < 1.0:
            raise ValueError(f"target_rate must lie in (0, 1), got {target_rate}")
        self.pool = pool
        self.head_input = head_input
        self.routing_threshold = float(routing_threshold)

        self.readout_q = nn.Parameter(torch.randn(d_model) / math.sqrt(d_model))

        self.prob_activation = prob_activation
        self.fixed_input_rate = fixed_input_rate
        self.input_rate = input_rate
        self.input_type = input_type

        self.input_dim = int(input_dim)

        if input_type == "token":
            if vocab_size is None:
                raise ValueError("token input requires vocab_size")
            self.inp = nn.Embedding(vocab_size, d_model)
        elif input_type == "continuous":
            self.inp = nn.Linear(self.input_dim, d_model)
        else:
            raise ValueError(f"Unsupported input_type={input_type}")

        # Binary spike input encoder for flat Spike-Mamba.
        self.in_prob = nn.Linear(d_model, d_model, bias=True)
        self.in_spike = SpikeSamplerSurrogate(
            mode=spike_mode,
            sample_inference=False,
            threshold=self.routing_threshold,
        )

        b = math.log(target_rate) - math.log(1.0 - target_rate)
        nn.init.constant_(self.in_prob.bias, b)
        self.in_prob.bias.requires_grad_(False)

        blocks = []
        for i in range(n_layers):
            core = core_cls(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                spiking=False,
                spike_mode=spike_mode,
                prob_activation=prob_activation,
                sample_inference=False,
                threshold=self.routing_threshold,
                layer_idx=i,
            )
            if hasattr(core, "prob_head") and getattr(core.prob_head, "bias", None) is not None:
                nn.init.constant_(core.prob_head.bias, b)
            blocks.append(ResidualPreNorm(d_model, core))

        self.blocks = nn.ModuleList(blocks)
        self.norm = get_norm(d_model)
        self.head = nn.Linear(d_model, num_classes)

        self.last_rate = None
        self.last_input_rate = None
        self.last_maxchan = None
        self.last_p0 = None
        self.last_sgc_loss = None

        self.enable_sgc = False
        self.sgc_layer_ids = []

    def pool_tokens(self, feats: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        # feats: [B, L, D]
        if self.pool == "last":
            return feats[:, -1, :]
        elif self.pool == "mean":
            if lengths is not None:
                positions = torch.arange(feats.size(1), device=feats.device).unsqueeze(0)
                mask = positions < lengths.to(feats.device).unsqueeze(1)
                return (feats * mask.unsqueeze(-1)).sum(dim=1) / lengths.to(feats.device).clamp_min(1).unsqueeze(-1)
            return feats.mean(dim=1)
        elif self.pool == "attn":
            score = torch.matmul(feats, self.readout_q)          # [B, L]
            weight = torch.softmax(score, dim=1).unsqueeze(-1)   # [B, L, 1]
            return (feats * weight).sum(dim=1)
        else:
            raise ValueError(f"Unsupported pool={self.pool}")

    def _encode_input(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.input_type == "token":
            if x.dim() != 2:
                raise ValueError(f"Token input expects shape (B, L), got {tuple(x.shape)}")
            v = self.inp(x.long())
        elif self.input_type == "continuous":
            if x.dim() == 2:
                v = self.inp(x.float().unsqueeze(-1))
            elif x.dim() == 3:
                v = self.inp(x.float())
            else:
                raise ValueError(f"Continuous LRA input expects [B,L] or [B,L,C], got {tuple(x.shape)}")
        else:
            raise ValueError(f"Unsupported input_type={self.input_type}")

        if self.prob_activation == "sigmoid":
            p0 = torch.sigmoid(self.in_prob(v))
        else:
            p0 = torch.clamp(self.in_prob(v), 0.0, 1.0)

        if self.fixed_input_rate:
            thresh = torch.quantile(p0.detach().flatten(), 1.0 - self.input_rate)
            s_hard = (p0 >= thresh).to(dtype=p0.dtype)
            s = s_hard.detach() + p0 - p0.detach()
        else:
            s = self.in_spike(p0)

        self.last_p0 = p0
        self.last_input_rate = s.detach().float().mean()
        return s, v

    def encode_sequence(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        s, v = self._encode_input(x)

        self.last_sgc_loss = None

        for blk in self.blocks:
            s, v = blk(s, v)

        if self.enable_sgc:
            sgc_terms = []
            for i in self.sgc_layer_ids:
                blk_loss = getattr(self.blocks[i], "last_sgc_loss", None)
                if blk_loss is not None:
                    sgc_terms.append(blk_loss)
            if len(sgc_terms) > 0:
                self.last_sgc_loss = sum(sgc_terms) / len(sgc_terms)

        # Final classifier input:
        #   spike: original behavior, classify from the final spike sequence s
        #   cont : classify from the continuous membrane/state sequence v
        #   both : simple fusion of spike and continuous states
        if self.head_input == "spike":
            out = self.norm(s)
        elif self.head_input == "cont":
            out = self.norm(v)
        elif self.head_input == "both":
            out = self.norm(s + v)
        else:
            raise ValueError(f"Unsupported head_input={self.head_input}")
        return self.pool_tokens(out, lengths=lengths)

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        feats = self.encode_sequence(x)
        logits = self.head(feats)

        if return_aux:
            sgc = self.last_sgc_loss
            if sgc is None:
                sgc = logits.new_zeros(())
            return logits, sgc

        return logits


class SpikeMambaNet(SequenceClassifierBase):
    def __init__(
        self,
        d_model: int = 128,
        n_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        spike_mode: str = "prob",
        prob_activation: str = "sigmoid",
        residual: bool = True,
        pool: str = "last",
        target_rate: float = 0.6,
        routing_threshold: float = 0.5,
        fixed_input_rate: bool = False,
        input_rate: float = 0.05,
        num_classes: int = 10,
        input_type: str = "continuous",
        vocab_size: Optional[int] = None,
        input_dim: int = 1,
        head_input: str = "spike",
    ):
        super().__init__(
            d_model=d_model,
            n_layers=n_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            spike_mode=spike_mode,
            prob_activation=prob_activation,
            pool=pool,
            target_rate=target_rate,
            routing_threshold=routing_threshold,
            fixed_input_rate=fixed_input_rate,
            input_rate=input_rate,
            num_classes=num_classes,
            input_type=input_type,
            vocab_size=vocab_size,
            core_cls=Mamba,
            input_dim=input_dim,
            head_input=head_input,
        )


class PairRetrievalModel(nn.Module):
    def __init__(self, encoder: nn.Module, d_model: int, num_classes: int = 2):
        super().__init__()
        self.encoder = encoder

        # Same NLI interaction as S4-main's RetrievalHead.
        self.head = nn.Sequential(
            nn.Linear(d_model * 4, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_classes),
        )

        self.last_rate = None
        self.last_maxchan = None
        self.last_sgc_loss = None

    def forward(self, x, return_aux: bool = False):
        if not isinstance(x, (tuple, list)) or len(x) not in (2, 4):
            raise ValueError("Retrieval task expects x=(doc1, doc2, lengths1, lengths2)")

        x1, x2 = x[:2]
        lengths1, lengths2 = x[2:] if len(x) == 4 else (None, None)

        h1 = self.encoder.encode_sequence(x1, lengths=lengths1)
        sgc1 = getattr(self.encoder, "last_sgc_loss", None)

        h2 = self.encoder.encode_sequence(x2, lengths=lengths2)
        sgc2 = getattr(self.encoder, "last_sgc_loss", None)

        if sgc1 is not None and sgc2 is not None:
            self.last_sgc_loss = 0.5 * (sgc1 + sgc2)
        elif sgc1 is not None:
            self.last_sgc_loss = sgc1
        elif sgc2 is not None:
            self.last_sgc_loss = sgc2
        else:
            self.last_sgc_loss = None

        pair = torch.cat([h1, h2, h1 - h2, h1 * h2], dim=-1)
        logits = self.head(pair)

        if return_aux:
            sgc = self.last_sgc_loss
            if sgc is None:
                sgc = logits.new_zeros(())
            return logits, sgc

        return logits


def move_batch_to_device(x, device: str):
    if isinstance(x, (tuple, list)):
        return type(x)(move_batch_to_device(v, device) for v in x)
    if not torch.is_tensor(x):
        return x
    if x.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.long):
        return x.to(device=device, dtype=torch.long)
    return x.to(device=device, dtype=torch.float32)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", str(s)).strip("_") or "run"


def _reproducible_args(args: argparse.Namespace) -> Dict[str, Any]:
    """Return checkpoint metadata without local paths or user labels."""
    private_keys = {
        "data_dir",
        "hf_local_path",
        "ssm_imdb_cache_dir",
        "ssm_listops_cache_dir",
        "save_dir",
        "record_dir",
        "run_name",
    }
    return {key: value for key, value in vars(args).items() if key not in private_keys}


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def setup_model_for_multi_gpu(model: nn.Module, device: str, max_gpus: int = 2) -> nn.Module:
    model = model.to(device)

    if device == "cuda" and torch.cuda.is_available():
        n_visible = torch.cuda.device_count()
        use_n = min(max_gpus, n_visible)
        if use_n >= 2:
            device_ids = list(range(use_n))
            print(f"[INFO] Using nn.DataParallel on GPUs {device_ids}")
            model = nn.DataParallel(model, device_ids=device_ids)
        else:
            print("[INFO] Single GPU visible, keep single-GPU training.")
    return model


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    device: str,
    apply_sgc_loss: bool = False,
    sgc_lambda: float = 0.1,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    label_smoothing: float = 0.0,
) -> Tuple[float, float]:
    model.train()
    total_ce = 0.0
    total_loss = 0.0

    for x, y in loader:
        x = move_batch_to_device(x, device)
        y = y.to(device=device)
        opt.zero_grad(set_to_none=True)

        if apply_sgc_loss:
            logits, L_sgc = model(x, return_aux=True)
            if not torch.is_tensor(L_sgc):
                L_sgc = logits.new_zeros(())
            if L_sgc.ndim > 0:
                L_sgc = L_sgc.mean()
        else:
            logits = model(x)
            L_sgc = logits.new_zeros(())

        L_ce = F.cross_entropy(logits, y, label_smoothing=label_smoothing)
        loss = L_ce + (sgc_lambda * L_sgc if apply_sgc_loss else 0.0)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if scheduler is not None:
            scheduler.step()

        total_ce += L_ce.item() * y.size(0)
        total_loss += loss.item() * y.size(0)

    avg_ce = total_ce / len(loader.dataset)
    avg_loss = total_loss / len(loader.dataset)
    return avg_ce, avg_loss

def _routing_rate_snapshot(model: nn.Module) -> Optional[Dict[str, Any]]:
    """Return input/generated/scan routing rates from the most recent forward."""
    base = model.module if isinstance(model, nn.DataParallel) else model
    encoder = getattr(base, "encoder", base)
    input_rate = getattr(encoder, "last_input_rate", None)
    blocks = getattr(encoder, "blocks", None)
    if input_rate is None or blocks is None:
        return None

    layer_rates = [getattr(block, "last_rate", None) for block in blocks]
    if any(rate is None for rate in layer_rates):
        return None

    input_value = float(input_rate.detach().item())
    generated_values = [float(rate.detach().item()) for rate in layer_rates]
    # Block 0 scans the input routing state; later blocks scan the state generated
    # by the preceding block. The last generated state is used only by the head.
    scan_values = [input_value, *generated_values[:-1]]
    return {
        "input": input_value,
        "generated_layers": generated_values,
        "scan_layers": scan_values,
    }


def eval_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    label_smoothing: float = 0.0,
) -> Tuple[float, float, Optional[Dict[str, Any]]]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_n = 0
    rate_weight = 0
    input_rate_sum = 0.0
    generated_rate_sums = None
    scan_rate_sums = None
    with torch.no_grad():
        for x, y in loader:
            x = move_batch_to_device(x, device)
            y = y.to(device=device)
            logits = model(x)
            loss = F.cross_entropy(logits, y, label_smoothing=label_smoothing)
            total_loss += loss.item() * y.size(0)
            total_correct += (logits.argmax(dim=-1) == y).sum().item()
            total_n += y.size(0)
            snapshot = _routing_rate_snapshot(model)
            if snapshot is not None:
                batch_n = y.size(0)
                input_rate_sum += snapshot["input"] * batch_n
                if generated_rate_sums is None:
                    generated_rate_sums = [0.0] * len(snapshot["generated_layers"])
                    scan_rate_sums = [0.0] * len(snapshot["scan_layers"])
                for i, value in enumerate(snapshot["generated_layers"]):
                    generated_rate_sums[i] += value * batch_n
                for i, value in enumerate(snapshot["scan_layers"]):
                    scan_rate_sums[i] += value * batch_n
                rate_weight += batch_n

    routing_stats = None
    if rate_weight > 0:
        generated_rates = [value / rate_weight for value in generated_rate_sums]
        scan_rates = [value / rate_weight for value in scan_rate_sums]
        routing_stats = {
            "input_rate": input_rate_sum / rate_weight,
            "generated_layer_rates": generated_rates,
            "scan_layer_rates": scan_rates,
            "mean_scan_rate": sum(scan_rates) / len(scan_rates),
        }
    return total_loss / total_n, total_correct / total_n, routing_stats


def smoke_test_training_step(
    model: nn.Module,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    opt: torch.optim.Optimizer,
    device: str,
    label_smoothing: float = 0.0,
) -> None:
    model.train()
    x, y = next(iter(train_loader))
    x = move_batch_to_device(x, device)
    y = y.to(device=device)
    opt.zero_grad(set_to_none=True)
    logits = model(x)
    loss = F.cross_entropy(logits, y, label_smoothing=label_smoothing)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

    model.eval()
    with torch.no_grad():
        x, y = next(iter(eval_loader))
        x = move_batch_to_device(x, device)
        y = y.to(device=device)
        logits = model(x)
        eval_loss = F.cross_entropy(logits, y, label_smoothing=label_smoothing)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    else:
        peak_mb = 0.0
    print(
        f"[SMOKE] ok | train_loss={loss.item():.4f} | "
        f"eval_loss={eval_loss.item():.4f} | peak_alloc={peak_mb:.1f} MiB"
    )




# -----------------------------------------------------------------------------
def build_model(args, num_classes: int, vocab_size: Optional[int]) -> nn.Module:
    """Build the Spike Mamba classifier for an LRA task."""
    if args.input_type == "continuous" and args.input_mode == "poisson":
        raise ValueError("Spike-Mamba does not use poisson input mode; use --input_mode cont.")

    common = dict(
        d_model=args.d_model,
        n_layers=args.n_layers,
        d_state=args.d_state,
        d_conv=args.d_conv,
        expand=args.expand,
        pool=args.pool,
        num_classes=num_classes,
        input_type=args.input_type,
        vocab_size=vocab_size,
        input_dim=1,
        head_input=args.head_input,
    )
    encoder = SpikeMambaNet(
        **common,
        spike_mode=args.spike_mode,
        prob_activation=args.prob_activation,
        residual=bool(args.residual),
        target_rate=args.target_rate,
        routing_threshold=args.routing_threshold,
        fixed_input_rate=bool(args.fixed_input_rate),
        input_rate=args.input_rate,
    )
    model_label = "Spike Mamba (flat compressed scan, s/v dual state)"
    model = PairRetrievalModel(encoder, args.d_model, num_classes=num_classes) if args.task == "retrieval" else encoder
    print(f"{model_label}{' (retrieval)' if args.task == 'retrieval' else ''} params: {count_params(model):,}")
    return model


def build_loaders(args):
    valid_loader = None

    if args.task == "image":
        if args.s4_image_pipeline:
            args.input_type = "continuous"
            args.input_mode = "cont"
            args.max_length = 32 * 32
            train_loader, valid_loader, test_loader = build_s4_cifar10_image_loaders(
                args.data_dir,
                args.batch_size,
                val_split=args.s4_image_val_split,
                seed=args.s4_image_split_seed,
                num_workers=args.image_workers,
                aug=args.s4_image_aug,
            )
            print(
                f"[INFO] S4 CIFAR image pipeline ON | grayscale=True | "
                f"normalize=(122.6/255, 61.0/255) | seq_len={args.max_length} | "
                f"val_split={args.s4_image_val_split:g} | aug={args.s4_image_aug}"
            )
        else:
            train_loader, test_loader = build_cifar10_image_loaders(
                args.data_dir,
                args.batch_size,
                input_mode=args.input_mode,
                T=args.T,
            )
        num_classes = 10
        vocab_size = None

    elif args.task == "text":
        if args.input_mode != "cont":
            print(f"[WARN] task={args.task} uses token loader. input_mode={args.input_mode} is ignored.")
        if args.text_pipeline == "ssm_benchmark":
            train_loader, test_loader, num_classes, vocab_size = build_ssm_benchmark_imdb_loaders(
                data_dir=args.data_dir,
                batch_size=args.batch_size,
                cache_dir=args.ssm_imdb_cache_dir,
            )
            args.max_length = 4096
        else:
            train_loader, test_loader, num_classes, vocab_size = build_text_loaders(
                data_dir=args.data_dir,
                batch_size=args.batch_size,
                max_length=args.max_length,
                hf_dataset_name=args.hf_dataset_name,
                hf_config_name=args.hf_config_name,
                hf_train_split=args.hf_train_split,
                hf_test_split=args.hf_test_split,
                pad_id=args.pad_id,
                hf_local_path=args.hf_local_path,
            )

    elif args.task == "listops":
        if args.input_mode != "cont":
            print(f"[WARN] task={args.task} uses token loader. input_mode={args.input_mode} is ignored.")
        if args.listops_pipeline == "ssm_benchmark":
            train_loader, test_loader, num_classes, vocab_size = build_ssm_benchmark_listops_loaders(
                data_dir=args.data_dir,
                batch_size=args.batch_size,
                cache_dir=args.ssm_listops_cache_dir,
            )
            args.max_length = 2048
        else:
            train_loader, test_loader, num_classes, vocab_size = build_listops_loaders(
                data_dir=args.data_dir,
                batch_size=args.batch_size,
                max_length=args.max_length,
                hf_dataset_name=args.hf_dataset_name,
                hf_config_name=args.hf_config_name,
                hf_train_split=args.hf_train_split,
                hf_test_split=args.hf_test_split,
                pad_id=args.pad_id,
                hf_local_path=args.hf_local_path,
            )

    elif args.task == "retrieval":
        if args.input_mode != "cont":
            print(f"[WARN] task={args.task} uses token pair loader. input_mode={args.input_mode} is ignored.")
        train_loader, test_loader, num_classes, vocab_size = build_retrieval_loaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            max_length=args.max_length,
            hf_dataset_name=args.hf_dataset_name,
            hf_config_name=args.hf_config_name,
            hf_train_split=args.hf_train_split,
            hf_test_split=args.hf_test_split,
            pad_id=args.pad_id,
            hf_local_path=args.hf_local_path,
        )

    elif args.task == "pathfinder":
        args.input_type = "continuous" if args.pathfinder_continuous else "token"
        args.max_length = args.pathfinder_resolution ** 2
        if args.pad_id != 0:
            print(f"[WARN] Pathfinder tokenization reserves pad_id=0, overriding pad_id {args.pad_id} -> 0")
            args.pad_id = 0
        if args.input_type == "token" and args.input_mode != "cont":
            print(f"[WARN] task={args.task} tokenizes Pathfinder images directly. input_mode={args.input_mode} is ignored.")
        train_loader, test_loader = build_pathfinder_loaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            input_mode="cont",
            T=args.T,
            pathfinder_resolution=args.pathfinder_resolution,
            pathfinder_level=args.pathfinder_level,
            input_type=args.input_type,
            pathfinder_bins=args.pathfinder_bins,
            pad_id=args.pad_id,
            hf_local_path=args.hf_local_path,
            pathfinder_split_seed=args.pathfinder_split_seed,
            pathfinder_center=bool(args.pathfinder_center),
        )
        num_classes = 2
        vocab_size = None if args.input_type == "continuous" else args.pathfinder_bins + 1

    else:
        raise ValueError(f"Unsupported task={args.task}")

    if args.num_classes is not None:
        num_classes = args.num_classes
    if args.vocab_size is not None:
        vocab_size = args.vocab_size

    return train_loader, valid_loader, test_loader, num_classes, vocab_size


def main():
    ap = argparse.ArgumentParser(description="Train Spike Mamba on LRA tasks.")
    ap.add_argument("--task", choices=["image", "text", "listops", "retrieval", "pathfinder"], default="listops")

    ap.add_argument("--data_dir", type=str, default="./data")
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--d_model", type=int, default=None)
    ap.add_argument("--n_layers", type=int, default=None)
    ap.add_argument("--input_type", choices=["continuous", "token"], default=None)
    ap.add_argument("--max_length", type=int, default=None)
    ap.add_argument("--num_classes", type=int, default=None)
    ap.add_argument("--vocab_size", type=int, default=None)

    ap.add_argument("--input_mode", choices=["cont", "bernoulli", "poisson"], default="cont")
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--pool", choices=["last", "mean", "attn"], default=None)
    ap.add_argument("--head_input", choices=["spike", "cont", "both"], default="spike")
    ap.add_argument("--s4_image_pipeline", type=int, default=0)
    ap.add_argument("--s4_image_val_split", type=float, default=0.1)
    ap.add_argument("--s4_image_split_seed", type=int, default=42)
    ap.add_argument("--s4_image_aug", choices=["none", "crop_flip", "crop_flip_erasing", "crop_flip_affine"], default="none")
    ap.add_argument("--image_workers", type=int, default=0)

    ap.add_argument("--d_state", type=int, default=None)
    ap.add_argument("--d_conv", type=int, default=4)
    ap.add_argument("--expand", type=int, default=None)
    ap.add_argument("--spike_mode", choices=["ste", "prob"], default="ste")
    ap.add_argument("--prob_activation", choices=["sigmoid", "clamp"], default="sigmoid")
    ap.add_argument("--residual", type=int, default=1)
    ap.add_argument("--target_rate", type=float, default=0.3)
    ap.add_argument(
        "--routing_threshold",
        type=float,
        default=0.5,
        help="Hard probability threshold used by every binary routing head.",
    )
    ap.add_argument("--fixed_input_rate", type=int, default=0)
    ap.add_argument("--input_rate", type=float, default=0.1)

    ap.add_argument("--hf_dataset_name", type=str, default="SaylorTwift/long-range-arena")
    ap.add_argument("--hf_config_name", type=str, default=None)
    ap.add_argument("--hf_train_split", type=str, default="train")
    ap.add_argument("--hf_test_split", type=str, default="test")
    ap.add_argument("--pad_id", type=int, default=0)
    ap.add_argument("--hf_local_path", type=str, default=None)
    ap.add_argument("--text_pipeline", choices=["native", "ssm_benchmark"], default="native",
                    help="Text preprocessing: native byte tokens, or the cached character-level IMDb pipeline from ssm-benchmark.")
    ap.add_argument("--ssm_imdb_cache_dir", type=str, default=None,
                    help="Optional explicit path to the processed ssm-benchmark IMDb cache directory.")
    ap.add_argument("--listops_pipeline", choices=["native", "ssm_benchmark"], default="native",
                    help="ListOps preprocessing: native TSV tokenization, or the exact cached ssm-benchmark pipeline.")
    ap.add_argument("--ssm_listops_cache_dir", type=str, default=None,
                    help="Optional explicit path to the processed ssm-benchmark ListOps cache directory.")

    ap.add_argument("--pathfinder_resolution", type=int, default=32)
    ap.add_argument("--pathfinder_level", choices=["easy", "intermediate", "hard", "all"], default="easy")
    ap.add_argument("--pathfinder_split_seed", type=int, default=123)
    ap.add_argument("--pathfinder_bins", type=int, default=256)
    ap.add_argument("--pathfinder_continuous", type=int, default=1,
                    help="Use the continuous Pathfinder-32 input used by the ssm-benchmark configuration.")
    ap.add_argument("--pathfinder_center", type=int, default=0)

    ap.add_argument("--label_smoothing", type=float, default=0.0)

    ap.add_argument("--use_scheduler", type=int, default=0)
    ap.add_argument("--ssm_benchmark_schedule", type=int, default=0,
                    help="Use ssm-benchmark's epoch-wise cosine schedule (eta_min=5e-6).")
    ap.add_argument("--warmup_steps", type=int, default=0)
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--sgc_lambda", type=float, default=0.1)
    ap.add_argument("--smoke_test", type=int, default=0)

    ap.add_argument("--save_dir", type=str, default="checkpoints")
    ap.add_argument("--save_best_only", type=int, default=1)
    ap.add_argument("--save_unique_best", type=int, default=1)
    ap.add_argument("--record_dir", type=str, default="experiment_records")
    ap.add_argument("--run_name", type=str, default=None)
    ap.add_argument("--target_acc", type=float, default=95.6)

    args = ap.parse_args()
    args.model = "spike_mamba"

    if args.task in ("listops", "text", "retrieval") and (args.hf_config_name is None or str(args.hf_config_name).lower() in ("", "none", "null")):
        args.hf_config_name = args.task

    args = apply_task_defaults(args)

    if args.task == "pathfinder":
        args.input_type = "continuous" if args.pathfinder_continuous else "token"
        args.max_length = args.pathfinder_resolution ** 2
        args.vocab_size = None if args.pathfinder_continuous else args.pathfinder_bins + 1
        if args.pool is None or args.pool == "last":
            args.pool = "mean"

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"Device: {device}")
    if device == "cuda":
        print(f"Visible CUDA GPUs: {n_gpus}")
    print(f"Task: {args.task}")
    print(f"Model: {args.model}")
    print(f"input_type={args.input_type} input_mode={args.input_mode} pool={args.pool}")
    print(
        f"routing_threshold={args.routing_threshold:g} "
        f"target_initial_rate={args.target_rate:g}"
    )

    train_loader, valid_loader, test_loader, num_classes, vocab_size = build_loaders(args)
    model = build_model(args, num_classes=num_classes, vocab_size=vocab_size)
    model = setup_model_for_multi_gpu(model, device, max_gpus=2)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    epoch_scheduler = None
    epoch_warmup = None

    if args.ssm_benchmark_schedule:
        import pytorch_warmup as warmup
        epoch_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.epochs, eta_min=5e-6
        )
        epoch_warmup = (
            warmup.LinearWarmup(opt, warmup_period=args.warmup_steps)
            if args.warmup_steps > 0 else None
        )
        scheduler = None
        print(
            "Scheduler: ssm-benchmark compatible - epoch-wise cosine "
            f"T_max={args.epochs}, eta_min=5e-6, warmup={args.warmup_steps} epochs"
        )
    elif args.use_scheduler:
        warmup_steps = int(args.warmup_ratio * total_steps) if args.warmup_ratio > 0.0 else args.warmup_steps
        print(f"Scheduler: ON - warmup={warmup_steps} steps, cosine decay over {total_steps} total steps")

        def _lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / max(1, warmup_steps)
            progress = float(current_step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=_lr_lambda)
    else:
        print(f"Scheduler: OFF - constant lr={args.lr:.2e} for {total_steps} total steps")
        scheduler = None

    best = 0.0
    os.makedirs(args.save_dir, exist_ok=True)
    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = _safe_name(args.run_name or f"{args.task}_{args.model}_seed{args.seed}_{run_stamp}")
    reproducible_args = _reproducible_args(args)
    record_path = Path(args.record_dir) / f"{args.task}_{args.model}_best.jsonl"
    print(f"[INFO] run_name={run_name}")
    print(f"[INFO] best records -> {record_path}")
    print(f"[INFO] label_smoothing={args.label_smoothing:g}")
    eval_loader = valid_loader if valid_loader is not None else test_loader
    eval_name = "val" if valid_loader is not None else "test"

    if args.task == "retrieval" and args.smoke_test == 2:
        model.train()
        x, y = next(iter(train_loader))
        x = move_batch_to_device(x, device)
        y = y.to(device)

        opt_dbg = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)

        for step in range(300):
            opt_dbg.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()

            total_grad = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_grad += float(p.grad.detach().abs().mean().cpu())

            opt_dbg.step()

            if step % 20 == 0:
                acc = (logits.argmax(-1) == y).float().mean().item()
                print(
                    f"[OVERFIT] step={step} "
                    f"loss={loss.item():.4f} "
                    f"acc={acc*100:.2f}% "
                    f"logit_std={logits.std().item():.4f} "
                    f"grad_mean_sum={total_grad:.6e}",
                    flush=True,
                )
        return

    if args.task == "retrieval" and args.smoke_test == 3:
        base = model.module if hasattr(model, "module") else model
        base.eval()

        x, y = next(iter(train_loader))
        x = move_batch_to_device(x, device)
        y = y.to(device)
        if not isinstance(x, (tuple, list)) or len(x) != 2:
            raise ValueError("fixed-head retrieval overfit expects x=(doc1, doc2)")

        with torch.no_grad():
            h1 = base.encoder.encode_sequence(x[0]).detach()
            h2 = base.encoder.encode_sequence(x[1]).detach()
            pair_h = torch.cat([h1, h2], dim=-1).detach()

        print(
            "[FIXED] "
            f"h1_std={h1.std().item():.6f} h2_std={h2.std().item():.6f} "
            f"pair_h_std={pair_h.std().item():.6f}",
            flush=True,
        )
        print(f"[FIXED] y={y.detach().cpu().tolist()}", flush=True)

        current_head = copy.deepcopy(base.head).to(device)
        opt_current = torch.optim.AdamW(current_head.parameters(), lr=1e-2, weight_decay=0.0)

        for step in range(500):
            opt_current.zero_grad(set_to_none=True)
            logits = current_head(pair_h)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt_current.step()

            if step % 20 == 0:
                acc = (logits.argmax(-1) == y).float().mean().item()
                print(
                    f"[FIXED_CURRENT_HEAD] step={step} "
                    f"loss={loss.item():.4f} "
                    f"acc={acc*100:.2f}% "
                    f"logit_std={logits.std().item():.4f}",
                    flush=True,
                )

        rich_head = nn.Sequential(
            nn.Linear(pair_h.shape[-1], args.d_model),
            nn.ReLU(),
            nn.Linear(args.d_model, max(2, args.d_model // 2)),
            nn.ReLU(),
            nn.Linear(max(2, args.d_model // 2), num_classes),
        ).to(device)
        opt_rich = torch.optim.AdamW(rich_head.parameters(), lr=1e-2, weight_decay=0.0)

        for step in range(500):
            opt_rich.zero_grad(set_to_none=True)
            logits = rich_head(pair_h)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt_rich.step()

            if step % 20 == 0:
                acc = (logits.argmax(-1) == y).float().mean().item()
                print(
                    f"[FIXED_RICH_HEAD] step={step} "
                    f"loss={loss.item():.4f} "
                    f"acc={acc*100:.2f}% "
                    f"logit_std={logits.std().item():.4f}",
                    flush=True,
                )
        return

    if args.smoke_test:
        smoke_test_training_step(
            model,
            train_loader,
            eval_loader,
            opt,
            device,
            label_smoothing=args.label_smoothing,
        )
        return

    for epoch in range(1, args.epochs + 1):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        epoch_t0 = time.perf_counter()

        tr_ce, tr_loss = train_one_epoch(
            model,
            train_loader,
            opt,
            device,
            apply_sgc_loss=False,
            sgc_lambda=args.sgc_lambda,
            scheduler=scheduler,
            label_smoothing=args.label_smoothing,
        )

        if epoch_scheduler is not None:
            if epoch_warmup is None:
                epoch_scheduler.step()
            else:
                with epoch_warmup.dampening():
                    epoch_scheduler.step()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            train_peak_mib = torch.cuda.max_memory_allocated() / 1024**2
        else:
            train_peak_mib = 0.0
        train_epoch_sec = time.perf_counter() - epoch_t0

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        eval_t0 = time.perf_counter()
        te_loss, te_acc, routing_stats = eval_one_epoch(
            model,
            eval_loader,
            device,
            label_smoothing=args.label_smoothing,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            eval_peak_mib = torch.cuda.max_memory_allocated() / 1024**2
        else:
            eval_peak_mib = 0.0
        eval_epoch_sec = time.perf_counter() - eval_t0

        is_best = te_acc > best
        if is_best:
            best = te_acc

        current_lr = scheduler.get_last_lr()[0] if scheduler is not None else args.lr
        state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        save_payload = {
            "epoch": epoch,
            "model_state_dict": state_dict,
            "optimizer_state_dict": opt.state_dict(),
            "args": reproducible_args,
            "best_acc": best,
            "run_name": run_name,
        }

        latest_ckpt = os.path.join(args.save_dir, f"{args.task}_{args.model}_last.pt")
        if not args.save_best_only:
            torch.save(save_payload, latest_ckpt)

        best_mark = ""
        if is_best:
            best_ckpt = os.path.join(args.save_dir, f"{args.task}_{args.model}_best.pt")
            torch.save(save_payload, best_ckpt)
            unique_best_ckpt = None
            if args.save_unique_best:
                unique_best_ckpt = os.path.join(
                    args.save_dir,
                    f"{run_name}_best.pt",
                )
                torch.save(save_payload, unique_best_ckpt)

            record = {
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "run_name": run_name,
                "epoch": epoch,
                "task": args.task,
                "model": args.model,
                "eval_split": eval_name,
                "eval_loss": te_loss,
                "eval_acc": te_acc,
                "eval_acc_percent": te_acc * 100.0,
                "best_acc": best,
                "best_acc_percent": best * 100.0,
                "train_ce": tr_ce,
                "train_loss": tr_loss,
                "lr": current_lr,
                "routing_threshold": args.routing_threshold,
                "routing_stats": routing_stats,
                "checkpoint": Path(best_ckpt).name,
                "unique_checkpoint": Path(unique_best_ckpt).name if unique_best_ckpt else None,
                "args": reproducible_args,
            }
            _append_jsonl(record_path, record)
            manifest_path = Path(args.save_dir) / f"{run_name}_best_manifest.json"
            manifest_path.write_text(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
            best_mark = f" ok saved -> {unique_best_ckpt or best_ckpt}"
            if args.target_acc > 0.0 and te_acc * 100.0 >= args.target_acc:
                best_mark += f" TARGET_REACHED({args.target_acc:.2f}%)"

        rate_summary = ""
        if routing_stats is not None:
            layer_text = ",".join(f"{rate:.4f}" for rate in routing_stats["scan_layer_rates"])
            rate_summary = (
                f" | mean_scan_rate {routing_stats['mean_scan_rate']:.4f}"
                f" | scan_rates [{layer_text}]"
            )

        print(
            f"Epoch {epoch:02d} | "
            f"lr {current_lr:.2e} | "
            f"train_time {train_epoch_sec:.3f}s | "
            f"eval_time {eval_epoch_sec:.3f}s | "
            f"train_peak_alloc {train_peak_mib:.1f}MiB | "
            f"eval_peak_alloc {eval_peak_mib:.1f}MiB | "
            f"train_loss {tr_loss:.4f} | "
            f"{eval_name}_loss {te_loss:.4f} | "
            f"{eval_name}_acc {te_acc*100:.2f}% | best {best*100:.2f}%{best_mark}"
            f"{rate_summary}"
        )

    print("Done.")


if __name__ == "__main__":
    main()
