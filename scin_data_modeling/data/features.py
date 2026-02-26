"""Feature engineering utilities for the SCIN dataset.

This module contains feature transformations extracted from the preprocessing
pipeline: checkbox normalization, demographic parsing, leakage column removal,
and one-hot encoding. It is designed to run *after* the cleaning step in
:mod:`scin_data_modeling.data.preprocess`.
"""

from __future__ import annotations

import re

import pandas as pd

CHECKBOX_PREFIXES = (
    "race_ethnicity_",
    "textures_",
    "body_parts_",
    "condition_symptoms_",
    "other_symptoms_",
)

LEAKAGE_COLUMNS = {
    "dermatologist_skin_condition_on_label_name",
    "dermatologist_skin_condition_confidence",
    "dermatologist_gradable_for_skin_condition_1",
    "dermatologist_gradable_for_skin_condition_2",
    "dermatologist_gradable_for_skin_condition_3",
    "dermatologist_gradable_for_fitzpatrick_skin_type_1",
    "dermatologist_gradable_for_fitzpatrick_skin_type_2",
    "dermatologist_gradable_for_fitzpatrick_skin_type_3",
    "dermatologist_fitzpatrick_skin_type_label_1",
    "dermatologist_fitzpatrick_skin_type_label_2",
    "dermatologist_fitzpatrick_skin_type_label_3",
    "gradable_for_monk_skin_tone_india",
    "gradable_for_monk_skin_tone_us",
    "monk_skin_tone_label_india",
    "monk_skin_tone_label_us",
}


# ── Parsing helpers ────────────────────────────────────────────────────────────


def parse_age_group_start(value: str | int | float | None) -> float | None:
    """Extract the lower bound from an age-group string (e.g. ``'age_18-29'`` → ``18.0``)."""
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"", "nan", "<na>", "none", "age_unknown"}:
        return None
    if text.startswith("age_"):
        text = text[4:]
    match = re.search(r"\d+", text)
    if match:
        return float(match.group(0))
    return None


def parse_fitzpatrick_ordinal(value: str | int | float | None) -> float | None:
    """Map Fitzpatrick skin type labels to ordinal floats (``'FST1'`` → ``1.0``)."""
    if value is None:
        return None
    text = str(value).strip().upper()
    if text in {"", "NAN", "<NA>", "NONE"}:
        return None
    mapping = {
        "FST1": 1.0,
        "FST2": 2.0,
        "FST3": 3.0,
        "FST4": 4.0,
        "FST5": 5.0,
        "FST6": 6.0,
        "NONE_SELECTED": None,
    }
    return mapping.get(text)


# ── Feature engineering ────────────────────────────────────────────────────────


def _checkbox_columns(columns: list[str]) -> list[str]:
    return [c for c in columns if any(c.startswith(p) for p in CHECKBOX_PREFIXES)]


def _safe_sum(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    if not columns:
        return pd.Series(0, index=df.index)
    return df[columns].sum(axis="columns")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering transformations to a cleaned DataFrame.

    Transformations include:
    - Checkbox columns normalized to 0/1 integers
    - Count features: ``n_textures``, ``n_body_parts``, ``n_condition_symptoms``,
      ``n_other_symptoms``
    - Conflict flags for contradictory symptom responses
    - Fitzpatrick ordinal encoding
    - Age-group lower-bound extraction
    - Sex-at-birth ``OTHER_OR_UNSPECIFIED`` flag
    """
    out = df.copy()
    checkbox_cols = _checkbox_columns(out.columns.tolist())

    for col in checkbox_cols:
        if col in out.columns:
            out[col] = out[col].notna().astype("int8")

    texture_cols = [c for c in checkbox_cols if c.startswith("textures_")]
    body_part_cols = [c for c in checkbox_cols if c.startswith("body_parts_")]
    cond_cols = [
        c
        for c in checkbox_cols
        if c.startswith("condition_symptoms_") and c != "condition_symptoms_no_relevant_experience"
    ]
    other_cols = [
        c
        for c in checkbox_cols
        if c.startswith("other_symptoms_") and c != "other_symptoms_no_relevant_symptoms"
    ]

    out["n_textures"] = _safe_sum(out, texture_cols).astype("int16")
    out["n_body_parts"] = _safe_sum(out, body_part_cols).astype("int16")
    out["n_condition_symptoms"] = _safe_sum(out, cond_cols).astype("int16")
    out["n_other_symptoms"] = _safe_sum(out, other_cols).astype("int16")

    if "condition_symptoms_no_relevant_experience" in out.columns:
        out["condition_symptoms_conflict_flag"] = (
            (out["condition_symptoms_no_relevant_experience"] == 1) & (out["n_condition_symptoms"] > 0)
        ).astype("int8")

    if "other_symptoms_no_relevant_symptoms" in out.columns:
        out["other_symptoms_conflict_flag"] = (
            (out["other_symptoms_no_relevant_symptoms"] == 1) & (out["n_other_symptoms"] > 0)
        ).astype("int8")

    if "fitzpatrick_skin_type" in out.columns:
        fst_upper = out["fitzpatrick_skin_type"].fillna("").astype(str).str.strip().str.upper()
        out["fitzpatrick_skin_type_none_selected_flag"] = (fst_upper == "NONE_SELECTED").astype("int8")
        out["fitzpatrick_skin_type_ordinal"] = out["fitzpatrick_skin_type"].map(parse_fitzpatrick_ordinal)

    if "age_group" in out.columns:
        age_upper = out["age_group"].fillna("").astype(str).str.strip().str.upper()
        out["age_group_unknown_flag"] = (age_upper == "AGE_UNKNOWN").astype("int8")
        out["age_group_start"] = out["age_group"].map(parse_age_group_start)

    if "sex_at_birth" in out.columns:
        sex_upper = out["sex_at_birth"].fillna("").astype(str).str.strip().str.upper()
        out["sex_at_birth_other_or_unspecified_flag"] = (sex_upper == "OTHER_OR_UNSPECIFIED").astype("int8")
        out["sex_at_birth"] = out["sex_at_birth"].mask(sex_upper == "OTHER_OR_UNSPECIFIED", pd.NA)

    return out


def drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns that leak dermatologist labels or grading info."""
    drop_cols = [c for c in LEAKAGE_COLUMNS if c in df.columns]
    return df.drop(columns=drop_cols)
