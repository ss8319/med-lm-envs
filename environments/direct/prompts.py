from __future__ import annotations

from typing import Dict, List


SECTION_LABELS = {
    "input1": "Chief Complaint",
    "input2": "History of Present Illness",
    "input3": "Past Medical History",
    "input4": "Family History",
    "input5": "Physical Exam",
    "input6": "Pertinent Results",
}


def build_note_prompt(input_content: Dict[str, str]) -> str:
    parts = []
    for key in (
        "input1",
        "input2",
        "input3",
        "input4",
        "input5",
        "input6",
    ):
        text = input_content.get(key)
        if text:
            parts.append(f"{SECTION_LABELS[key]}:\n{text}".rstrip())
    return "\n\n".join(parts)


def build_open_diagnosis_prompt(note: str) -> str:
    try:
        from DiReCT.utils.data_extraction import gen_disease_open  # type: ignore

        return gen_disease_open(note)
    except Exception:
        return (
            "Suppose you are a medical expert. Review the clinical note and output ONLY the final diagnosis.\n\n"
            f"Note:\n{note}\nYour 'Response':"
        )


def build_category_prompt(note: str, disease_options: List[str]) -> str:
    try:
        from DiReCT.utils.data_extraction import gen_disease_diagnose  # type: ignore

        return gen_disease_diagnose(note, disease_options)
    except Exception:
        options = ", ".join(disease_options)
        return (
            "Suppose you are a medical expert. Read the clinical note and pick exactly ONE disease from the list.\n"
            f"Options: {options}\n\nNote:\n{note}\nYour 'Response':"
        )


def build_observation_prompt(note: str, disease: str, premise: str | None = None) -> str:
    try:
        from DiReCT.utils.data_extraction import gen_disease_open2  # type: ignore

        return gen_disease_open2(note)
    except Exception:
        base = (
            "Suppose you are a medical expert. Review the clinical note and extract observations supporting the diagnosis.\n"
            f"Diagnosis: {disease}\n"
            "Return a Python list of [Observation, Reason, Disease] pairs."
        )
        if premise:
            base += f"\nPremise:\n{premise}"
        return base + f"\n\nNote:\n{note}\nYour 'Response':"


def build_flowchart_prompt(
    note: str,
    disease_category: str,
    leaf_candidates: List[str],
    observations_json: str,
    premise: str | None = None,
) -> str:
    leaf_str = ", ".join(leaf_candidates) if leaf_candidates else ""
    prompt = (
        "Suppose you are a medical expert. Using the observations extracted from the note, "
        "walk through the diagnostic flowchart for the disease category and output the diagnostic chain as a JSON list.\n"
        f"Disease category: {disease_category}\n"
        f"Observations (JSON): {observations_json}\n"
    )
    if leaf_str:
        prompt += f"Possible final diagnoses: {leaf_str}\n"
    if premise:
        prompt += f"Premise:\n{premise}\n"
    prompt += (
        "Return a JSON array of disease names in the order you reach them, starting from the most general node and ending with the final diagnosis.\n"
        "Do not include any additional explanation.\n\nNote:\n"
        f"{note}\nYour 'Response':"
    )
    return prompt


def build_pipeline_prompt(
    note: str,
    disease_category: str,
    leaf_candidates: List[str],
    include_observations: bool = True,
) -> str:
    leaf_str = ", ".join(leaf_candidates) if leaf_candidates else ""
    prompt = (
        "Suppose you are a medical expert. Perform the full DiReCT diagnostic pipeline on the clinical note.\n"
        "Return a JSON object with the following keys: 'category' (the disease category),\n"
        "'observations' (a list of [Observation, Reason, Disease] pairs), and 'chain' (the diagnostic chain as an ordered list).\n"
        "Do not include any extra keys or text.\n"
        f"Disease category hint: {disease_category}\n"
    )
    if leaf_str:
        prompt += f"Possible final diagnoses: {leaf_str}\n"
    if include_observations:
        prompt += "Ensure observations are drawn directly from the note text.\n"
    prompt += f"\nNote:\n{note}\nYour 'Response':"
    return prompt
