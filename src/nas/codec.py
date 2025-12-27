from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np

from src.evaluator import Candidate
from src.nas.utils import bounds_from_space, load_search_space


@dataclass
class Codec:
    space: Dict[str, Any]
    lb: np.ndarray
    ub: np.ndarray
    base_image_size: int

    @classmethod
    def from_yaml(cls, path: str | None) -> "Codec":
        space = load_search_space(path) if path else load_search_space("configs/search_space.yaml")
        lb, ub = bounds_from_space(space)
        base_image_size = int(space.get("base_image_size", 224))
        return cls(space=space, lb=lb, ub=ub, base_image_size=base_image_size)

    def clip_vec(self, vec: np.ndarray) -> np.ndarray:
        return np.clip(vec.astype(np.float32), self.lb, self.ub)

    def vec_to_candidate(self, vec: np.ndarray) -> Candidate:
        vec = self.clip_vec(vec)
        return Candidate.from_vector(vec)

    def json_to_vec(self, payload: Dict[str, Any]) -> np.ndarray:
        depth = float(payload.get("depth_mult", self.space["depth_mult"]["min"]))
        width = float(payload.get("width_mult", self.space["width_mult"]["min"]))
        res = float(payload.get("res_mult", self.space["res_mult"]["min"]))
        vec = np.array([depth, width, res], dtype=np.float32)
        return self.clip_vec(vec)

    def vec_to_json(self, vec: np.ndarray) -> Dict[str, Any]:
        cand = self.vec_to_candidate(vec)
        image_size = int(max(32, round(self.base_image_size * cand.res_mult)))
        return {
            "depth_mult": cand.depth_mult,
            "width_mult": cand.width_mult,
            "res_mult": cand.res_mult,
            "image_size": image_size,
        }

    def to_ofa_config(self, vec: np.ndarray) -> Dict[str, Any]:
        j = self.vec_to_json(vec)
        return {
            "net_id": "ofa_mbv3_d234_e346_k357_w1.0",
            "depth_mult": j["depth_mult"],
            "width_mult": j["width_mult"],
            "image_size": j["image_size"],
        }

