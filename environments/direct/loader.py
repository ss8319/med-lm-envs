from __future__ import annotations

import json
import logging
import os
import sys
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Support both package and module execution contexts
try:  # type: ignore
    from .config import DirectConfig  # type: ignore
    from .prompts import (  # type: ignore
        build_note_prompt,
        build_open_diagnosis_prompt,
        build_category_prompt,
        build_observation_prompt,
        build_flowchart_prompt,
        build_pipeline_prompt,
    )
except Exception:  # pragma: no cover - fallback for non-package import
    from config import DirectConfig  # type: ignore
    from prompts import (  # type: ignore
        build_note_prompt,
        build_open_diagnosis_prompt,
        build_category_prompt,
        build_observation_prompt,
        build_flowchart_prompt,
        build_pipeline_prompt,
    )

logger = logging.getLogger(__name__)

DIRECT_ROOT = Path(__file__).resolve().parent / "DiReCT"
if DIRECT_ROOT.exists():
    parent_str = str(DIRECT_ROOT.parent)
    if parent_str not in sys.path:
        sys.path.insert(0, parent_str)

DATA_ROOT = DIRECT_ROOT.parent / "mimic-iv-ext-direct-1.0.0" / "samples" / "Finished"


@contextmanager
def _temp_cwd(path: Path):
    prev = Path.cwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(prev)


def iter_sample_paths(root: Path = DATA_ROOT) -> List[Path]:
    if not root.exists():
        logger.warning("[DiReCT] data root missing", extra={"root": str(root)})
        return []
    return sorted(root.rglob("*.json"))


@lru_cache(maxsize=1)
def load_resources() -> Dict[str, Any]:
    resources: Dict[str, Any] = {}
    if not DIRECT_ROOT.exists():
        return resources

    try:
        from DiReCT.utils.data_analysis import (  # type: ignore
            disease_category,
            get_non_dict_keys,
            extract_keys,
        )
    except Exception as exc:
        logger.error("[DiReCT] failed to load resources", exc_info=exc)
        return resources

    with _temp_cwd(DIRECT_ROOT):
        disease_options, flowchart = disease_category()
    resources["disease_options"] = disease_options
    resources["flowchart"] = flowchart

    leaf_lookup: Dict[str, List[str]] = {}
    disease_lists: Dict[str, List[str]] = {}
    knowledge_lookup: Dict[str, Any] = {}
    for category in disease_options:
        flows = (flowchart.get(category, {}) or {}).get("diagnostic", {})
        leaf_lookup[category] = get_non_dict_keys(flows, "") if isinstance(flows, dict) else []
        disease_lists[category] = extract_keys(flows, "") if isinstance(flows, dict) else []
        knowledge_lookup[category] = (flowchart.get(category, {}) or {}).get("knowledge", {})
    resources["leaf_lookup"] = leaf_lookup
    resources["disease_lists"] = disease_lists
    resources["knowledge_lookup"] = knowledge_lookup

    return resources


def _load_json_top_key(path: Path) -> Optional[str]:
    try:
        with path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
    except Exception as exc:
        logger.error("[DiReCT] failed to load JSON", exc_info=exc, extra={"path": str(path)})
        return None

    for key in data.keys():
        if not key.lower().startswith("input"):
            return key.split("$", 1)[0]
    return None


def map_direct_sample(
    path: Path,
    *,
    config: DirectConfig,
    resources: Dict[str, Any],
    include_prompt: bool = True,
) -> Optional[Dict[str, Any]]:
    try:
        from DiReCT.utils.data_analysis import cal_a_json  # type: ignore
    except Exception as exc:
        logger.error("[DiReCT] utils not available", exc_info=exc)
        return None

    try:
        record_node, input_content, chain = cal_a_json(str(path))
    except Exception as exc:
        logger.error("[DiReCT] cal_a_json failed", exc_info=exc, extra={"path": str(path)})
        return None

    # use already-imported build_note_prompt from module imports

    note = build_note_prompt(input_content)
    gold = _load_json_top_key(path)
    if not gold and chain:
        gold = chain[-1]
    if not gold:
        gold = path.parent.name

    # infer disease category from directory structure
    try:
        disease_category = path.parent.parent.name
    except IndexError:
        disease_category = None

    disease_options = resources.get("disease_options", [])

    stage = getattr(config, "stage", "final")

    info: Dict[str, Any] = {
        "note": note,
        "chain": chain,
        "path": str(path),
        "disease_category": disease_category,
        "final_diagnosis": gold,
        "stage": stage,
    }

    if stage == "category":
        if include_prompt:
            if disease_options:
                question = build_category_prompt(note, disease_options)
            else:
                question = build_open_diagnosis_prompt(note)
        else:
            question = note
        answer = disease_category or gold
        return {
            "prompt": [{"role": "user", "content": question}],
            "answer": answer,
            "info": info,
        }

    if stage in {"observation", "flowchart"}:
        try:
            from DiReCT.utils.data_analysis import deduction_assemble  # type: ignore
        except Exception as exc:
            logger.error("[DiReCT] deduction assemble unavailable", exc_info=exc)
            return None

        obs_map = deduction_assemble(record_node)
        obs_json = json.dumps(obs_map, ensure_ascii=False)

    if stage == "observation":

        disease_for_prompt = disease_category or gold
        premise = None
        if config.include_premise_initial:
            try:
                from DiReCT.utils.data_analysis import combine_premise  # type: ignore

                knowledge_lookup = resources.get("knowledge_lookup", {})
                disease_lists = resources.get("disease_lists", {})
                knowledge = knowledge_lookup.get(disease_category or "", {})
                disease_list = disease_lists.get(disease_category or "", [])
                if knowledge and disease_list:
                    premise = combine_premise(knowledge, disease_list, initial=True)
            except Exception:
                premise = None

        question = (
            build_observation_prompt(note, disease_for_prompt, premise)
            if include_prompt
            else note
        )

        info["ground_truth_observations_json"] = obs_json

        return {
            "prompt": [{"role": "user", "content": question}],
            "answer": obs_json,
            "info": info,
        }

    if stage == "flowchart":
        observations = obs_json if 'obs_json' in locals() else json.dumps({})
        leaf_candidates = resources.get("leaf_lookup", {}).get(disease_category or "", [])
        premise = None
        if config.include_premise_iterative:
            try:
                from DiReCT.utils.data_analysis import combine_premise  # type: ignore

                knowledge_lookup = resources.get("knowledge_lookup", {})
                knowledge = knowledge_lookup.get(disease_category or "", {})
                if knowledge and leaf_candidates:
                    premise = combine_premise(knowledge, leaf_candidates)
            except Exception:
                premise = None

        question = (
            build_flowchart_prompt(
                note,
                disease_category or gold,
                leaf_candidates,
                observations,
                premise,
            )
            if include_prompt
            else note
        )

        info["ground_truth_observations_json"] = observations
        info["leaf_candidates"] = leaf_candidates

        return {
            "prompt": [{"role": "user", "content": question}],
            "answer": json.dumps(info.get("chain", []), ensure_ascii=False),
            "info": info,
        }

    # default final diagnosis stage
    if include_prompt:
        question = build_open_diagnosis_prompt(note)
    else:
        question = note

    return {
        "prompt": [{"role": "user", "content": question}],
        "answer": gold,
        "info": info,
    }


def build_dataset(config: DirectConfig) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    paths = iter_sample_paths()
    if config.disease_filter:
        df = config.disease_filter.lower()
        paths = [p for p in paths if df in str(p.parent).lower()]

    if config.max_examples != -1:
        paths = paths[: max(0, config.max_examples)]

    logger.info(
        "[DiReCT] loading samples",
        extra={"direct_config": config.to_dict(), "num_paths": len(paths)},
    )

    resources = load_resources()

    rows: List[Dict[str, Any]] = []
    for path in paths:
        mapped = map_direct_sample(path, config=config, resources=resources)
        if mapped is None:
            continue
        mapped.setdefault("info", {})
        mapped["info"]["config"] = config.to_dict()
        if resources and getattr(config, "stage", "final") != "category":
            category = mapped["info"].get("disease_category")
            if category:
                mapped["info"]["leaf_candidates"] = resources.get("leaf_lookup", {}).get(category, [])
        mapped["info"]["resources_available"] = bool(resources)
        rows.append(mapped)
    return rows, resources
