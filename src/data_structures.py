"""
Defines the core data structures used in the Bibliography Ranker.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from torch import Tensor

@dataclass
class Paper:
    """
    Represents a research paper.
    """
    title: str
    abstract: str
    openalex_id: str = ""
    embedding: Optional[Tensor] = None
    similarity: float = 0.0
