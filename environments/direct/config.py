from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class DirectConfig:
    mode: str = "open"  # open-diagnosis (default) vs closed (use disease list)
    use_premise: bool = False  # whether to inject KG premises (future checkpoints)
    max_examples: int = -1
    disease_filter: Optional[str] = None
    seed: int = 42
    strict_direct_normalization: bool = True
    treat_synonyms: bool = False
    stage: str = "final"  # "category" for Stage A
    include_premise_initial: bool = False
    include_premise_iterative: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "use_premise": self.use_premise,
            "max_examples": self.max_examples,
            "disease_filter": self.disease_filter,
            "seed": self.seed,
            "strict_direct_normalization": self.strict_direct_normalization,
            "treat_synonyms": self.treat_synonyms,
        }
