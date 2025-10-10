from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from .DiReCT.statistics import statistic_one_pred  # type: ignore
    from .DiReCT.utils.data_analysis import (  # type: ignore
        cal_a_json,
        deduction_assemble,
        get_all_file_paths,
    )
except Exception:  # pragma: no cover - fallback when executed as script
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from DiReCT.statistics import statistic_one_pred  # type: ignore
    from DiReCT.utils.data_analysis import (  # type: ignore
        cal_a_json,
        deduction_assemble,
        get_all_file_paths,
    )


def _load_results(results_path: Path) -> List[Dict]:
    entries: List[Dict] = []
    with results_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            if not line.strip():
                continue
            entries.append(json.loads(line))
    return entries


def _build_eval_record(entry: Dict) -> Tuple[Dict, Dict]:
    info = entry.get("info", {})
    gold_chain = info.get("chain", [])
    gold_path = info.get("path")
    if not gold_path:
        raise ValueError("Missing original sample path in results entry")

    record_node, _, _ = cal_a_json(gold_path)
    gold_obs = deduction_assemble(record_node)

    completion = entry.get("completion") or []
    if completion and isinstance(completion, list):
        completion_text = completion[-1].get("content", "")
    else:
        completion_text = entry.get("completion", "")

    try:
        parsed = json.loads(completion_text)
    except Exception:
        parsed = {}

    pipeline_obs = parsed.get("observations", {}) if isinstance(parsed, dict) else {}
    pipeline_chain = parsed.get("chain", []) if isinstance(parsed, dict) else []

    predicate_obs = {}
    if isinstance(pipeline_obs, dict):
        for obs, value in pipeline_obs.items():
            if isinstance(value, list) and len(value) == 3:
                predicate_obs[obs] = value

    paired = {}
    for idx, obs in enumerate(predicate_obs.keys()):
        if obs in gold_obs:
            gold = gold_obs[obs]
            pred = predicate_obs[obs]
            paired[str((idx, idx))] = [gold[2], pred[2], gold[0], pred[1], "Yes" if gold[0] == pred[1] else "No"]

    eval_record = {
        "chain_gt": gold_chain,
        "chain_pred": pipeline_chain,
        "len_ob_gt": len(gold_obs),
        "len_ob_pred": len(predicate_obs),
        "ob_record_paired": paired,
    }

    return pipeline_obs, eval_record


def evaluate(results_path: Path) -> Dict[str, float]:
    entries = _load_results(results_path)

    metrics_accum = {
        "acc_cat": [],
        "acc_diag": [],
        "comp_pre": [],
        "comp_re": [],
        "comp_coverage": [],
        "faith_ob": [],
        "faith_all": [],
    }

    temp_eval_dir = results_path.parent / "direct_eval_tmp"
    temp_eval_dir.mkdir(exist_ok=True)

    for idx, entry in enumerate(entries):
        _, eval_record = _build_eval_record(entry)
        eval_file = temp_eval_dir / f"sample_{idx}.json"
        with eval_file.open("w", encoding="utf-8") as fp:
            json.dump(eval_record, fp, ensure_ascii=False, indent=2)

        acc_cat, acc_diag, comp_pre, comp_re, comp_cov, faith_ob, faith_all = statistic_one_pred(str(eval_file))

        metrics_accum["acc_cat"].append(acc_cat)
        metrics_accum["acc_diag"].append(acc_diag)
        metrics_accum["comp_pre"].append(comp_pre)
        metrics_accum["comp_re"].append(comp_re)
        metrics_accum["comp_coverage"].append(comp_cov)
        metrics_accum["faith_ob"].append(faith_ob)
        metrics_accum["faith_all"].append(faith_all)

    summary = {}
    for key, values in metrics_accum.items():
        if values:
            summary[key] = sum(values) / len(values)
        else:
            summary[key] = 0.0

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DiReCT metrics on vf-eval results")
    parser.add_argument("results", type=str, help="Path to results.jsonl produced by vf-eval")
    args = parser.parse_args()

    results_path = Path(args.results).resolve()
    if not results_path.exists():
        raise FileNotFoundError(results_path)

    summary = evaluate(results_path)
    print("=== DiReCT Metric Summary ===")
    for key, value in summary.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()

