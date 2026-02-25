"""Preprocessing utilities for the SCIN dataset.

This module is intentionally focused on data ingestion + cleaning. It avoids
feature engineering and modeling concerns.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd

RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")

# Demographic / feature columns to carry from scin_cases.csv into the splits.
CASE_FEATURE_COLS = [
    "age_group",
    "sex_at_birth",
    "fitzpatrick_skin_type",
    "race_ethnicity_american_indian_or_alaska_native",
    "race_ethnicity_asian",
    "race_ethnicity_black_or_african_american",
    "race_ethnicity_hispanic_latino_or_spanish_origin",
    "race_ethnicity_middle_eastern_or_north_african",
    "race_ethnicity_native_hawaiian_or_pacific_islander",
    "race_ethnicity_white",
    "race_ethnicity_other_race",
    "race_ethnicity_prefer_not_to_answer",
    "combined_race",
    "textures_raised_or_bumpy",
    "textures_flat",
    "textures_rough_or_flaky",
    "textures_fluid_filled",
    "body_parts_head_or_neck",
    "body_parts_arm",
    "body_parts_palm",
    "body_parts_back_of_hand",
    "body_parts_torso_front",
    "body_parts_torso_back",
    "body_parts_genitalia_or_groin",
    "body_parts_buttocks",
    "body_parts_leg",
    "body_parts_foot_top_or_side",
    "body_parts_foot_sole",
    "body_parts_other",
    "condition_symptoms_bothersome_appearance",
    "condition_symptoms_bleeding",
    "condition_symptoms_increasing_size",
    "condition_symptoms_darkening",
    "condition_symptoms_itching",
    "condition_symptoms_burning",
    "condition_symptoms_pain",
    "condition_symptoms_no_relevant_experience",
    "other_symptoms_fever",
    "other_symptoms_chills",
    "other_symptoms_fatigue",
    "other_symptoms_joint_pain",
    "other_symptoms_mouth_sores",
    "other_symptoms_shortness_of_breath",
    "other_symptoms_no_relevant_symptoms",
    "related_category",
    "condition_duration",
]


def load_combined_df(raw_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """Load and merge the SCIN cases and labels CSVs on case_id.

    Files are expected at ``<raw_dir>/dataset/scin_cases.csv`` and
    ``<raw_dir>/dataset/scin_labels.csv`` — the path structure preserved by
    :func:`scin_data_modeling.data.download.download_csvs`.
    """
    dataset_dir = raw_dir / "dataset"
    cases_df = pd.read_csv(dataset_dir / "scin_cases.csv")
    labels_df = pd.read_csv(dataset_dir / "scin_labels.csv")
    return pd.merge(cases_df, labels_df, on="case_id", how="inner")


def _parse_label_list(raw_value: object) -> list[str] | None:
    """Parse the ``dermatologist_skin_condition_on_label_name`` column.

    Values are stored as string representations of Python lists, e.g.
    ``"['Eczema', 'Psoriasis', 'Dermatitis']"``.  An empty list ``"[]"``
    or a missing value means the case has no label.

    Returns ``None`` when no label is available.
    """
    if pd.isna(raw_value):
        return None
    try:
        parsed = ast.literal_eval(str(raw_value))
    except (ValueError, SyntaxError):
        return None
    if not isinstance(parsed, list) or len(parsed) == 0:
        return None
    return parsed


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        clean_item = str(item).strip()
        if not clean_item or clean_item in seen:
            continue
        seen.add(clean_item)
        deduped.append(clean_item)
    return deduped


def _collect_image_paths(row: pd.Series) -> list[str]:
    """Return a list of 1-3 GCS blob paths for a case, skipping missing entries."""
    paths: list[str] = []
    for i in range(1, 4):
        p = row.get(f"image_{i}_path")
        if pd.notna(p):
            paths.append(str(p))
    return paths


def build_clean_df(
    combined_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build a cleaned DataFrame for downstream work.

    Columns in the returned DataFrame:
    - all original merged columns from ``scin_cases.csv`` + ``scin_labels.csv``
      except ``dermatologist_skin_condition_on_label_name``
    - image_paths : JSON-encoded list of 1-3 image paths
    - num_images  : image count per case
    - label_all   : JSON-encoded deduplicated condition list
    - label       : JSON-encoded first 3 conditions from ``label_all``
    """
    df = combined_df.copy()

    # Light string cleaning for object columns
    object_cols = df.select_dtypes(include="object").columns
    for col in object_cols:
        df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

    # Parse and clean labels; drop rows with no usable target
    df["label_all"] = df["dermatologist_skin_condition_on_label_name"].apply(_parse_label_list)
    df = df[df["label_all"].notna()].copy()
    df["label_all"] = df["label_all"].apply(_dedupe_keep_order)
    df = df[df["label_all"].apply(len) > 0].copy()
    df["label"] = df["label_all"].apply(lambda xs: xs[:3])

    # Collect image paths (1-3) as a JSON list
    df["image_paths"] = df.apply(_collect_image_paths, axis=1).apply(json.dumps)
    df["num_images"] = df["image_paths"].apply(lambda s: len(json.loads(s)))
    df = df[df["num_images"] > 0].copy()

    # JSON-encode labels so lists survive CSV round-trips
    df["label_all"] = df["label_all"].apply(json.dumps)
    df["label"] = df["label"].apply(json.dumps)

    # Keep all merged raw columns except the unparsed raw label string.
    keep_cols = [c for c in df.columns if c != "dermatologist_skin_condition_on_label_name"]

    return df[keep_cols].reset_index(drop=True)


def build_processed_df(
    combined_df: pd.DataFrame,
    raw_dir: Path = RAW_DATA_DIR,
) -> pd.DataFrame:
    """Backward-compatible alias for older code paths."""
    del raw_dir
    return build_clean_df(combined_df)


def create_train_test_split(
    processed_df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Random train/test split.

    Parameters
    ----------
    processed_df:
        Output of :func:`build_processed_df`.
    test_size:
        Fraction of data to reserve for the test set (default 0.2).
    random_state:
        Seed for reproducibility.

    Returns
    -------
    (train_df, test_df) with reset indices.
    """
    from sklearn.model_selection import train_test_split

    train_df, test_df = train_test_split(
        processed_df,
        test_size=test_size,
        random_state=random_state,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def save_splits(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    out_dir: Path = PROCESSED_DATA_DIR,
) -> None:
    """Save train/test DataFrames as CSVs under *out_dir*."""
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.csv"
    test_path = out_dir / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"Saved {len(train_df):,} training records  → {train_path}")
    print(f"Saved {len(test_df):,} test records       → {test_path}")


def save_clean_data(
    clean_df: pd.DataFrame,
    out_dir: Path = PROCESSED_DATA_DIR,
) -> Path:
    """Save the cleaned dataset as ``cleaned.csv`` under *out_dir*."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cleaned.csv"
    clean_df.to_csv(out_path, index=False)
    print(f"Saved {len(clean_df):,} cleaned records → {out_path}")
    return out_path


def load_split(split: str, processed_dir: Path = PROCESSED_DATA_DIR) -> pd.DataFrame:
    """Load a saved split CSV and deserialise JSON-encoded columns.

    Parameters
    ----------
    split:
        ``'train'`` or ``'test'``.
    """
    df = pd.read_csv(processed_dir / f"{split}.csv")
    df["image_paths"] = df["image_paths"].apply(json.loads)
    df["label"] = df["label"].apply(json.loads)
    return df


def load_clean_data(processed_dir: Path = PROCESSED_DATA_DIR) -> pd.DataFrame:
    """Load ``cleaned.csv`` and deserialize list-like JSON columns."""
    df = pd.read_csv(processed_dir / "cleaned.csv")
    if "image_paths" in df.columns:
        df["image_paths"] = df["image_paths"].apply(json.loads)
    if "label" in df.columns:
        df["label"] = df["label"].apply(json.loads)
    if "label_all" in df.columns:
        df["label_all"] = df["label_all"].apply(json.loads)
    return df
