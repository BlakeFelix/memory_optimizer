from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Set, Optional


@dataclass
class Memory:
    memory_id: str
    content: str
    timestamp: datetime
    type: str
    project_id: Optional[str] = None
    entities: Set[str] = field(default_factory=set)
    importance_weight: float = 0.0
    access_count: int = 0
