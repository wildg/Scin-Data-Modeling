from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

TARGET_COLUMN = "weighted_skin_condition_label"
GROUP_COLUMN = "case_id"

IMAGE_PATH_COLUMNS = ("image_1_path", "image_2_path", "image_3_path")
IMAGE_SHOT_TYPE_COLUMNS = ("image_1_shot_type", "image_2_shot_type", "image_3_shot_type")

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

ID_COLUMNS = ("case_id", "image_path")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df


def _parse_age_group_start(value: str | int | float | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {'', 'nan', '<na>', 'none'}:
        return None
    if not text:
        return None
    if text == "age_unknown":
        return None
    if text.startswith("age_"):
        text = text[4:]
    # Extract first number from patterns like "18-29", "65+", "0-2", etc.
    match = re.search(r"\d+", text)
    if match:
        return float(match.group(0))
    return None


def _parse_fitzpatrick_ordinal(value: str | int | float | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    if text in {'', 'NAN', '<NA>', 'NONE'}:
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
    if text in mapping:
        return mapping[text]
    return None


def reshape_cases_to_images(cases_df: pd.DataFrame) -> pd.DataFrame:
    base_cols = [
        col
        for col in cases_df.columns
        if col not in IMAGE_PATH_COLUMNS and col not in IMAGE_SHOT_TYPE_COLUMNS
    ]
    rows: list[pd.DataFrame] = []

    for idx in (1, 2, 3):
        path_col = f"image_{idx}_path"
        shot_col = f"image_{idx}_shot_type"
        if path_col not in cases_df.columns:
            continue

        subset_cols = base_cols + [path_col]
        if shot_col in cases_df.columns:
            subset_cols.append(shot_col)

        subset = cases_df[subset_cols].copy()
        subset = subset.rename(columns={path_col: "image_path", shot_col: "image_shot_type"})
        subset["image_index"] = idx
        rows.append(subset)

    if not rows:
        raise ValueError("No image path columns found in cases data.")

    image_df = pd.concat(rows, ignore_index=True)
    image_df = image_df.dropna(subset=["image_path"]).reset_index(drop=True)
    return image_df


def case_level_split(
    image_df: pd.DataFrame,
    *,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    case_targets = image_df[[GROUP_COLUMN, TARGET_COLUMN]].drop_duplicates()
    targets_per_case = case_targets.groupby(GROUP_COLUMN)[TARGET_COLUMN].nunique()
    if (targets_per_case > 1).any():
        raise ValueError("A case_id maps to multiple target labels; cannot stratify safely.")

    case_ids = case_targets[GROUP_COLUMN].tolist()
    stratify_labels = case_targets[TARGET_COLUMN].tolist()

    train_case_ids, test_case_ids = train_test_split(
        case_ids,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_labels,
    )

    train_df = image_df[image_df[GROUP_COLUMN].isin(train_case_ids)].copy()
    test_df = image_df[image_df[GROUP_COLUMN].isin(test_case_ids)].copy()

    overlap = set(train_df[GROUP_COLUMN]).intersection(set(test_df[GROUP_COLUMN]))
    if overlap:
        raise ValueError("Leakage detected: overlapping case_id values across train and test.")
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def _checkbox_columns(columns: list[str]) -> list[str]:
    return [c for c in columns if any(c.startswith(prefix) for prefix in CHECKBOX_PREFIXES)]


def _safe_sum(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    if not columns:
        return pd.Series(0, index=df.index)
    return df[columns].sum(axis="columns")


def _engineer_features(df: pd.DataFrame, checkbox_cols: list[str]) -> pd.DataFrame:
    engineered = df.copy()

    for col in checkbox_cols:
        if col in engineered.columns:
            engineered[col] = engineered[col].notna().astype("int8")

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

    engineered["n_textures"] = _safe_sum(engineered, texture_cols).astype("int16")
    engineered["n_body_parts"] = _safe_sum(engineered, body_part_cols).astype("int16")
    engineered["n_condition_symptoms"] = _safe_sum(engineered, cond_cols).astype("int16")
    engineered["n_other_symptoms"] = _safe_sum(engineered, other_cols).astype("int16")

    if "condition_symptoms_no_relevant_experience" in engineered.columns:
        engineered["condition_symptoms_conflict_flag"] = (
            (engineered["condition_symptoms_no_relevant_experience"] == 1)
            & (engineered["n_condition_symptoms"] > 0)
        ).astype("int8")

    if "other_symptoms_no_relevant_symptoms" in engineered.columns:
        engineered["other_symptoms_conflict_flag"] = (
            (engineered["other_symptoms_no_relevant_symptoms"] == 1)
            & (engineered["n_other_symptoms"] > 0)
        ).astype("int8")

    if "fitzpatrick_skin_type" in engineered.columns:
        fst_upper = engineered["fitzpatrick_skin_type"].fillna("").astype(str).str.strip().str.upper()
        engineered["fitzpatrick_skin_type_none_selected_flag"] = (fst_upper == "NONE_SELECTED").astype("int8")
        engineered["fitzpatrick_skin_type_ordinal"] = engineered["fitzpatrick_skin_type"].map(
            _parse_fitzpatrick_ordinal
        )
    if "age_group" in engineered.columns:
        age_upper = engineered["age_group"].fillna("").astype(str).str.strip().str.upper()
        engineered["age_group_unknown_flag"] = (age_upper == "AGE_UNKNOWN").astype("int8")
        engineered["age_group_start"] = engineered["age_group"].map(_parse_age_group_start)
    if "sex_at_birth" in engineered.columns:
        sex_upper = engineered["sex_at_birth"].fillna("").astype(str).str.strip().str.upper()
        engineered["sex_at_birth_other_or_unspecified_flag"] = (sex_upper == "OTHER_OR_UNSPECIFIED").astype("int8")
        engineered["sex_at_birth"] = engineered["sex_at_birth"].mask(
            sex_upper == "OTHER_OR_UNSPECIFIED",
            pd.NA,
        )
    return engineered


def _drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [c for c in LEAKAGE_COLUMNS if c in df.columns]
    return df.drop(columns=drop_cols)


def prepare_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train_df = _drop_leakage_columns(train_df)
    test_df = _drop_leakage_columns(test_df)

    checkbox_cols = _checkbox_columns(train_df.columns.tolist())
    train_df = _engineer_features(train_df, checkbox_cols)
    test_df = _engineer_features(test_df, checkbox_cols)

    metadata_cols = [c for c in ID_COLUMNS if c in train_df.columns]

    train_target = train_df[TARGET_COLUMN].copy()
    test_target = test_df[TARGET_COLUMN].copy()

    train_model = train_df.drop(columns=[TARGET_COLUMN])
    test_model = test_df.drop(columns=[TARGET_COLUMN])

    numeric_cols = [
        c for c in train_model.select_dtypes(include=["number", "bool"]).columns if c not in metadata_cols
    ]
    categorical_cols = [c for c in train_model.columns if c not in numeric_cols and c not in metadata_cols]

    for col in numeric_cols:
        median_value = train_model[col].median()
        if pd.isna(median_value):
            median_value = 0.0
        train_model[col] = train_model[col].fillna(median_value)
        test_model[col] = test_model[col].fillna(median_value)

    for col in categorical_cols:
        train_model[col] = train_model[col].fillna("Unknown").astype(str)
        test_model[col] = test_model[col].fillna("Unknown").astype(str)

    train_cat = pd.get_dummies(train_model[categorical_cols], prefix=categorical_cols, dtype="int8")
    test_cat = pd.get_dummies(test_model[categorical_cols], prefix=categorical_cols, dtype="int8")
    test_cat = test_cat.reindex(columns=train_cat.columns, fill_value=0)

    train_num = train_model[numeric_cols].reset_index(drop=True)
    test_num = test_model[numeric_cols].reset_index(drop=True)

    x_train = pd.concat([train_num, train_cat.reset_index(drop=True)], axis=1)
    x_test = pd.concat([test_num, test_cat.reset_index(drop=True)], axis=1)

    train_out = pd.concat(
        [train_df[metadata_cols].reset_index(drop=True), train_target.reset_index(drop=True), x_train],
        axis=1,
    )
    test_out = pd.concat(
        [test_df[metadata_cols].reset_index(drop=True), test_target.reset_index(drop=True), x_test],
        axis=1,
    )
    feature_columns = x_train.columns.tolist()
    return train_out, test_out, feature_columns


def run_pipeline(
    *,
    cases_path: Path,
    labels_path: Path,
    output_dir: Path,
    test_size: float,
    random_state: int,
) -> None:
    cases_df = _normalize_columns(pd.read_csv(cases_path))
    labels_df = _normalize_columns(pd.read_csv(labels_path))

    image_df = reshape_cases_to_images(cases_df)
    merged_df = image_df.merge(labels_df, on=GROUP_COLUMN, how="inner")

    if TARGET_COLUMN not in merged_df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' was not found after merge.")

    train_df, test_df = case_level_split(
        merged_df,
        test_size=test_size,
        random_state=random_state,
    )
    train_out, test_out, feature_columns = prepare_features(train_df, test_df)

    output_dir.mkdir(parents=True, exist_ok=True)
    train_out.to_csv(output_dir / "train_prepared.csv", index=False)
    test_out.to_csv(output_dir / "test_prepared.csv", index=False)

    manifest = {
        "cases_path": str(cases_path),
        "labels_path": str(labels_path),
        "target_column": TARGET_COLUMN,
        "group_column": GROUP_COLUMN,
        "random_state": random_state,
        "test_size": test_size,
        "train_rows": int(len(train_out)),
        "test_rows": int(len(test_out)),
        "train_cases": int(train_out[GROUP_COLUMN].nunique()),
        "test_cases": int(test_out[GROUP_COLUMN].nunique()),
        "feature_count": int(len(feature_columns)),
        "feature_columns": feature_columns,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Saved: {output_dir / 'train_prepared.csv'}")
    print(f"Saved: {output_dir / 'test_prepared.csv'}")
    print(f"Saved: {output_dir / 'manifest.json'}")
    print(f"Case leakage check passed: no overlapping '{GROUP_COLUMN}' values between train and test.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare SCIN cases and labels into leakage-safe train/test sets.")
    parser.add_argument("--cases-path", type=Path, default=Path("data/dataset_scin_cases.csv"))
    parser.add_argument("--labels-path", type=Path, default=Path("data/dataset_scin_labels.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(
        cases_path=args.cases_path,
        labels_path=args.labels_path,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()

