from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class TraceEntry:
    prompt: str
    response: str
    json_payload: Dict[str, Any]


@dataclass
class ExperienceBuffer:
    max_size: int = 128
    entries: List[TraceEntry] = field(default_factory=list)

    def add_if_improved(self, improved: bool, prompt: str, response: str, json_payload: Dict[str, Any]) -> bool:
        if not improved:
            return False
        self.entries.append(TraceEntry(prompt=prompt, response=response, json_payload=json_payload))
        if len(self.entries) > self.max_size:
            self.entries = self.entries[-self.max_size :]
        return True

    def pop_all(self) -> List[TraceEntry]:
        data = self.entries
        self.entries = []
        return data

    def __len__(self) -> int:
        return len(self.entries)


