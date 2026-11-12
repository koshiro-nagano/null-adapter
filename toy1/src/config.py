"""YAML config loader with single inheritance: a config may set `base: other.yaml`."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml


def _merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def load_config(path) -> dict:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    base = cfg.pop("base", None)
    return _merge(load_config(path.parent / base), cfg) if base else cfg
