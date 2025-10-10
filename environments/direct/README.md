# DiReCT Environment (MedARC)

MedARC’s `direct` environment packages the DiReCT (Diagnostic Reasoning for Clinical Notes) benchmark so it can be run with the Prime/Verifiers toolchain.

## Dataset

- **Source**: [PhysioNet – MIMIC-IV-Ext DiReCT 1.0.0](https://doi.org/10.13026/yf96-kc87)
- **Local layout**: `environments/direct/mimic-iv-ext-direct-1.0.0/`
- **Contents**: 511 annotated clinical notes (in `samples/Finished/**.json`) plus diagnostic knowledge graphs (`diagnostic_kg/`).

To download (PhysioNet login required):

```powershell
New-Item -ItemType Directory -Path environments/direct/mimic-iv-ext-direct-1.0.0 -Force | Out-Null
cd environments/direct/mimic-iv-ext-direct-1.0.0
wget -r -N -c -np --user <your_username> --ask-password https://physionet.org/files/mimic-iv-ext-direct/1.0.0/
```

## Installation

From the repository root:

```powershell
uv run vf-install direct
```

This installs the `direct` package (including DiReCT’s utilities) into the uv-managed virtual environment. Run it whenever `pyproject.toml` changes or after cloning the repo.

## Running Evaluations

Each stage of DiReCT can be run independently via the `stage` argument.

### Stage A – Disease Category

```powershell
$env:DIRECT_STAGE = "category"
uv run vf-eval direct -m gpt-4.1-mini -n 5 -r 1 -s
Remove-Item Env:DIRECT_STAGE
```

### Stage B – Observation Extraction

```powershell
$env:DIRECT_STAGE = "observation"
uv run vf-eval direct -m gpt-4.1-mini -n 5 -r 1 -s
Remove-Item Env:DIRECT_STAGE
```

### Stage C – Flowchart Reasoning

```powershell
$env:DIRECT_STAGE = "flowchart"
uv run vf-eval direct -m gpt-4.1-mini -n 5 -r 1 -s
Remove-Item Env:DIRECT_STAGE
```

Each run saves prompts/completions under `environments/direct/outputs/evals/direct--<model>/<run_id>/`.

### Combining Stages

Run stages sequentially (category → observation → flowchart) to produce all artifacts needed for official DiReCT metrics.

## DiReCT Metric Evaluation

After a `vf-eval direct ...` run, compute completeness/faithfulness scores using DiReCT’s own evaluation code:

```powershell
uv run python environments/direct/eval_metrics.py environments/direct/outputs/evals/direct--gpt-4.1-mini/<run_id>/results.jsonl
```

The script reuses `DiReCT/statistics.py` and prints:

- `acc_cat`, `acc_diag`
- `comp_pre`, `comp_re`, `comp_coverage`
- `faith_ob`, `faith_all`

Use `-h` to see options:

```powershell
uv run python environments/direct/eval_metrics.py -h
```

## Environment Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `stage` | str | `"final"` | Which evaluation stage to run (`"category"`, `"observation"`, `"flowchart"`). |
| `max_examples` | int | `-1` | Limit dataset size (use `-1` for full set). |
| `disease_filter` | str or null | `null` | Filter cases whose path includes this substring. |
| `include_premise_initial` | bool | `false` | Include knowledge-graph premises when prompting observation extraction. |
| `include_premise_iterative` | bool | `false` | Include iterative premises when prompting flowchart reasoning. |

## Notes

- The environment depends on DiReCT’s utilities (`DiReCT/utils/*`). Do not delete or move that folder.
- `eval_metrics.py` assumes run outputs follow the default Verifiers format.
- For multi-stage evaluation, run the stages in order and then call the metrics script on the final results.
- Checkpoints in this repo mirror DiReCT’s paper: category → observations → flowchart → metric aggregation.

