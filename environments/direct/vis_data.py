from pathlib import Path
import sys

from DiReCT.utils.data_analysis import cal_a_json, deduction_assemble


def resolve_sample_path() -> Path:
    # Allow override via CLI arg; otherwise use canonical path relative to this file
    if len(sys.argv) > 1:
        return Path(sys.argv[1]).resolve()

    base = Path(__file__).resolve().parent  # environments/direct
    return (
        base
        / "mimic-iv-ext-direct-1.0.0"
        / "samples"
        / "Finished"
        / "Acute Coronary Syndrome"
        / "NSTEMI"
        / "11535902-DS-14.json"
    )


def main() -> None:
    sample_path = resolve_sample_path()
    record_node, input_content, chain = cal_a_json(str(sample_path))

    # Minimal sanity output
    print(f"Loaded: {sample_path}")
    print(f"record_node: {len(record_node)} nodes")
    print(f"input_content keys: {list(input_content.keys())}")
    print(f"chain length: {len(chain)}")


if __name__ == "__main__":
    main()