from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import yaml

from src.evaluator import Candidate


def load_search_space(path: str | Path) -> Dict:
    config_path = Path(path)
    with config_path.open("r") as f:
        return yaml.safe_load(f)


def bounds_from_space(space: Dict) -> Tuple[np.ndarray, np.ndarray]:
    lb = np.array(
        [
            float(space["depth_mult"]["min"]),
            float(space["width_mult"]["min"]),
            float(space["res_mult"]["min"]),
        ],
        dtype=np.float32,
    )
    ub = np.array(
        [
            float(space["depth_mult"]["max"]),
            float(space["width_mult"]["max"]),
            float(space["res_mult"]["max"]),
        ],
        dtype=np.float32,
    )
    return lb, ub


def sample_candidate(space: Dict, rng: np.random.Generator | None = None) -> Candidate:
    rng = rng or np.random.default_rng()
    depth = rng.uniform(space["depth_mult"]["min"], space["depth_mult"]["max"])
    width = rng.uniform(space["width_mult"]["min"], space["width_mult"]["max"])
    res = rng.uniform(space["res_mult"]["min"], space["res_mult"]["max"])
    return Candidate(depth_mult=float(depth), width_mult=float(width), res_mult=float(res))


def vector_to_text(cand: Candidate, base_image_size: int = 224) -> str:
    res = int(base_image_size * cand.res_mult)
    return (
        f"A MobileNetV3 with depth multiplier {cand.depth_mult:.2f}, "
        f"width {cand.width_mult:.2f}, and resolution {res}."
    )


def append_rows_csv(path: str | Path, fieldnames: List[str], rows: Iterable[Dict]) -> None:
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_json(path: str | Path, payload: Dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2))

