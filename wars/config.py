import copy
from pathlib import Path

import yaml


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path, overrides=()):
    """Đọc YAML; hỗ trợ `base: <file>` để kế thừa và override dạng `train.epochs=50`."""
    path = Path(path)
    cfg = yaml.safe_load(path.read_text()) or {}
    if "base" in cfg:
        cfg = _merge(load_config(path.parent / cfg.pop("base")), cfg)
    for item in overrides:
        key, val = item.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(val)
    return cfg
