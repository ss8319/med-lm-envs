from __future__ import annotations

import logging
import json
import os
import re
from typing import Any, Dict, List, Optional

import verifiers as vf
from datasets import Dataset

# Support both package and module execution contexts
try:
    from .config import DirectConfig  # type: ignore
    from .loader import build_dataset  # type: ignore
except Exception:  # pragma: no cover - fallback for non-package import
    from config import DirectConfig  # type: ignore
    from loader import build_dataset  # type: ignore

# ---- Logging ----
logger = logging.getLogger(__name__)


# ---- Parsing utilities ----
def _get_text_from_completion(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return str(last.get("content", ""))
        return str(last)
    return str(completion)


def _normalize_diagnosis_custom(text: str) -> str:
    t = (text or "").lower()
    # Harmonize punctuation and dashes
    t = t.replace("–", "-").replace("—", "-")
    # Common expansions/collapses for MI-related terminology
    t = re.sub(r"myocardial\s+infarction", "mi", t)
    t = re.sub(r"st\s*-?\s*elevation", "ste", t)
    t = re.sub(r"non\s*-?\s*st\s*-?\s*elevation", "nste", t)
    # Strip non-alphanumerics
    t = re.sub(r"[^a-z0-9]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()

    # Heuristic canonicalization for common ACS variants
    tokens = t.split()
    token_set = set(tokens)
    if ("nste" in token_set or "nonst" in t or "non" in token_set) and ("mi" in token_set or "myocardial" in token_set):
        return "nstemi"
    if ("ste" in token_set or "st" in token_set) and ("mi" in token_set or "myocardial" in token_set):
        return "stemi"
    # Collapse spaces for simple exact compare
    return t.replace(" ", "")


def _normalize_category(text: str) -> str:
    return (text or "").strip()


def _parse_observation_completion(text: str) -> Dict[str, Any]:
    txt = (text or "").strip()
    try:
        data = json.loads(txt)
        if isinstance(data, list):
            observation_map = {}
            for item in data:
                if isinstance(item, list) and len(item) == 3:
                    observation_map[item[0]] = {
                        "reason": item[1],
                        "disease": item[2],
                    }
            return observation_map
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def _parse_flowchart_completion(text: str) -> List[str]:
    txt = (text or "").strip()
    try:
        data = json.loads(txt)
        if isinstance(data, list):
            return [str(item) for item in data]
    except Exception:
        pass
    return []


def _parse_pipeline_completion(text: str) -> Dict[str, Any]:
    txt = (text or "").strip()
    try:
        data = json.loads(txt)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def load_environment(
    max_examples: int = -1,
    disease_filter: Optional[str] = None,
    use_think: bool = False,
    strict_direct_normalization: bool = True,
    treat_synonyms: bool = False,
    mode: str = "open",
    use_premise: bool = False,
    seed: int = 42,
    stage: str = "final",
    include_premise_initial: bool = False,
    include_premise_iterative: bool = False,
    **kwargs,
) -> vf.Environment:
    """
    DiReCT v0.5 environment: final-diagnosis accuracy from clinical notes only.

    - Loads all Finished samples under MIMIC-IV-Ext-DiReCT
    - Builds a note string and asks model to output ONLY the diagnosis text
    - Scores accuracy with synonym-tolerant normalization for ACS variants
    """

    # Allow environment variable to override stage (handy when shells mangle JSON args)
    stage_env = os.getenv("DIRECT_STAGE")
    if stage_env:
        stage = stage_env

    config = DirectConfig(
        mode=mode,
        use_premise=use_premise,
        max_examples=max_examples,
        disease_filter=disease_filter,
        seed=seed,
        strict_direct_normalization=strict_direct_normalization,
        treat_synonyms=treat_synonyms,
        stage=stage,
        include_premise_initial=include_premise_initial,
        include_premise_iterative=include_premise_iterative,
    )

    rows, resources = build_dataset(config)
    ds = Dataset.from_list(rows) if rows else Dataset.from_list([])

    def _extract_answer_final(completion: Any) -> str:
        txt = _get_text_from_completion(completion)
        if config.strict_direct_normalization:
            try:
                from DiReCT.utils.data_analysis import capitalize_first_letter  # type: ignore

                return capitalize_first_letter(txt)
            except Exception:
                pass
        if config.treat_synonyms:
            return _normalize_diagnosis_custom(txt)
        return (txt or "").strip()

    def _extract_answer_category(completion: Any) -> str:
        txt = _get_text_from_completion(completion)
        try:
            from DiReCT.utils.data_analysis import capitalize_first_letter  # type: ignore

            return capitalize_first_letter(txt)
        except Exception:
            return _normalize_category(txt)

    def _extract_answer_observation(completion: Any) -> Dict[str, Any]:
        txt = _get_text_from_completion(completion)
        return _parse_observation_completion(txt)

    if config.stage == "category":
        parser = vf.Parser(extract_fn=_extract_answer_category)
    elif config.stage == "observation":
        parser = vf.Parser(extract_fn=_extract_answer_observation)
    elif config.stage == "flowchart":
        def _extract_flowchart(completion: Any) -> List[str]:
            return _parse_flowchart_completion(_get_text_from_completion(completion))

        parser = vf.Parser(extract_fn=_extract_flowchart)
    else:
        parser = vf.Parser(extract_fn=_extract_answer_final)

    def accuracy_reward(completion: Any, answer: str, **_kwargs) -> float:
        pred_raw = parser.parse_answer(completion)
        ans_raw = str(answer)

        if config.stage == "category":
            try:
                from DiReCT.utils.data_analysis import capitalize_first_letter  # type: ignore

                pred = capitalize_first_letter(pred_raw)
                gold = capitalize_first_letter(ans_raw)
            except Exception:
                pred = _normalize_category(pred_raw)
                gold = _normalize_category(ans_raw)
        elif config.stage == "observation":
            try:
                pred_map = pred_raw if isinstance(pred_raw, dict) else {}
                gold_map = json.loads(ans_raw)
            except Exception:
                gold_map = {}

            pred_keys = set(pred_map.keys())
            gold_keys = set(gold_map.keys())
            intersection = pred_keys & gold_keys
            matched = len(intersection)
            precision = matched / len(pred_keys) if pred_keys else 0.0
            recall = matched / len(gold_keys) if gold_keys else 0.0
            denom = (len(pred_keys) + len(gold_keys) - matched)
            coverage = matched / denom if denom else 0.0

            state = _kwargs.get("state")
            if isinstance(state, dict):
                metrics = state.setdefault("observation_metrics", {})
                metrics.update(
                    {
                        "matched": matched,
                        "total_pred": len(pred_keys),
                        "total_gold": len(gold_keys),
                        "precision": precision,
                        "recall": recall,
                        "coverage": coverage,
                    }
                )

            return coverage
        elif config.stage == "flowchart":
            try:
                pred_chain = pred_raw if isinstance(pred_raw, list) else []
                gold_chain = json.loads(ans_raw)
            except Exception:
                gold_chain = []

            if not gold_chain:
                return 0.0
            pred_normalized = [str(item).strip().lower() for item in pred_chain]
            gold_normalized = [str(item).strip().lower() for item in gold_chain]
            matches = sum(1 for p, g in zip(pred_normalized, gold_normalized) if p == g)
            precision = matches / len(pred_normalized) if pred_normalized else 0.0
            recall = matches / len(gold_normalized) if gold_normalized else 0.0
            denom = (len(pred_normalized) + len(gold_normalized) - matches)
            coverage = matches / denom if denom else 0.0

            state = _kwargs.get("state")
            if isinstance(state, dict):
                metrics = state.setdefault("flowchart_metrics", {})
                metrics.update(
                    {
                        "matched": matches,
                        "total_pred": len(pred_normalized),
                        "total_gold": len(gold_normalized),
                        "precision": precision,
                        "recall": recall,
                        "coverage": coverage,
                    }
                )

            return coverage
        else:
            if config.strict_direct_normalization:
                try:
                    from DiReCT.utils.data_analysis import capitalize_first_letter  # type: ignore

                    pred = capitalize_first_letter(pred_raw)
                    gold = capitalize_first_letter(ans_raw)
                except Exception:
                    pred = (pred_raw or "").strip()
                    gold = (ans_raw or "").strip()
            else:
                if config.treat_synonyms:
                    pred = _normalize_diagnosis_custom(pred_raw)
                    gold = _normalize_diagnosis_custom(ans_raw)
                else:
                    pred = (pred_raw or "").strip()
                    gold = (ans_raw or "").strip()

        return 1.0 if pred == gold else 0.0

    rubric = vf.Rubric(funcs=[accuracy_reward], weights=[1.0], parser=parser)

    # No explicit system prompt beyond what question embeds
    return vf.SingleTurnEnv(
        eval_dataset=ds,
        system_prompt=None,
        parser=parser,
        rubric=rubric,
        **kwargs,
    )

