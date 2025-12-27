from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Optional


class LatencyLUT:
    def __init__(self, path: Optional[Path] = None):
        self.path = path
        self.table: Dict[str, float] = {}
        if path and path.exists():
            self.table = json.loads(path.read_text())

    def get_latency(self, op_key: str) -> Optional[float]:
        return self.table.get(op_key)

    def sum_latency(self, ops: Dict[str, Any]) -> Optional[float]:
        # ops is expected to be a mapping of layer names -> op_key
        total = 0.0
        for _, key in ops.items():
            val = self.table.get(key)
            if val is None:
                return None
            total += float(val)
        return total


