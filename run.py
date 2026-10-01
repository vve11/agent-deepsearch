from dataclasses import dataclass, field
from typing import Literal
from uuid import uuid4


@dataclass
class ResearchRun:
    question: str

    id: str = field(default_factory=lambda: uuid4().hex)
    status: Literal["running", "completed", "failed"] = "running"
    answer: str | None = None
    evidence: list[dict[str, str]] = field(default_factory=list)
    error: str | None = None