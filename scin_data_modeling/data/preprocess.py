from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

TARGET_COLUMN = "weighted_skin_condition_label"
GROUP_COLUMN = "case_id"
TARGET_PROBS_COLUMN = "_target_probs"
TARGET_IS_LABELED_COLUMN = "_is_labeled"
TARGET_HARD_COLUMN = "_target_hard_label"
TARGET_PARSE_ERROR_COLUMN = "_target_parse_error"

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
    out = df.copy()
    out.columns = [c.strip() for c in out.columns]
    return out


def _parse_age_group_start(value: str | int | float | None) -> float | None:
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


def _parse_fitzpatrick_ordinal(value: str | int | float | None) -> float | None:
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
        "NONE_IDENTIFIED": None,
        "NONE_SELECTED": None,
    }
    return mapping.get(text)


def _parse_weighted_label(value: str | dict[str, object] | int | float | None) -> tuple[dict[str, float], int]:
    if value is None:
        return {}, 0
    text_value = str(value).strip()
    if text_value.lower() in {'', 'nan', '<na>', 'none'}:
        return {}, 0

    if isinstance(value, dict):
        raw = value
    else:
        text = str(value).strip()
        if text in {"", "{}"}:
            return {}, 0
        if not text.startswith("{"):
            return {text: 1.0}, 0
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return {}, 1
        if not isinstance(parsed, dict):
            return {}, 1
        raw = parsed

    cleaned: dict[str, float] = {}
    for key, weight in raw.items():
        label = str(key).strip()
        if not label:
            continue
        try:
            prob = float(str(weight))
        except (TypeError, ValueError):
            continue
        if prob > 0:
            cleaned[label] = prob

    if not cleaned:
        return {}, 0

    total = sum(cleaned.values())
    return {k: v / total for k, v in cleaned.items()}, 0


def _attach_target_metadata(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    parsed = out[TARGET_COLUMN].map(_parse_weighted_label)
    out[TARGET_PROBS_COLUMN] = parsed.map(lambda item: item[0])
    out[TARGET_PARSE_ERROR_COLUMN] = parsed.map(lambda item: item[1]).astype("int8")
    out[TARGET_IS_LABELED_COLUMN] = out[TARGET_PROBS_COLUMN].map(lambda d: int(bool(d)))
    out[TARGET_HARD_COLUMN] = out[TARGET_PROBS_COLUMN].map(lambda d: max(d, key=d.get) if d else pd.NA)
    return out


def _class_vocabulary(train_df: pd.DataFrame) -> list[str]:
    labels: set[str] = set()
    labeled = train_df[train_df[TARGET_IS_LABELED_COLUMN] == 1]
    for probs in labeled[TARGET_PROBS_COLUMN]:
        labels.update(probs.keys())
    return sorted(labels)


def _sanitize_label_token(label: str) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "_", label.strip()).strip("_").lower()
    return token or "empty_label"


def _build_target_column_map(class_vocab: list[str]) -> dict[str, str]:
    used: set[str] = set()
    mapping: dict[str, str] = {}
    for label in class_vocab:
        base = f"target_prob__{_sanitize_label_token(label)}"
        col = base
        idx = 2
        while col.lower() in used:
            col = f"{base}__{idx}"
            idx += 1
        used.add(col.lower())
        mapping[label] = col
    return mapping


def _build_target_frame(df: pd.DataFrame, class_vocab: list[str], target_column_map: dict[str, str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in df.iterrows():
        probs_raw = row[TARGET_PROBS_COLUMN]
        probs = probs_raw if isinstance(probs_raw, dict) else {}
        is_labeled = int(row[TARGET_IS_LABELED_COLUMN])
        target_hard_raw = row[TARGET_HARD_COLUMN]
        target_hard = target_hard_raw if is_labeled else pd.NA

        rec: dict[str, object] = {
            "target_raw": row[TARGET_COLUMN],
            "is_labeled": is_labeled,
            "target_parse_error": int(row[TARGET_PARSE_ERROR_COLUMN]),
            "target_hard_label": target_hard,
            "target_max_prob": max(probs.values()) if probs else pd.NA,
        }
        unseen_mass = 0.0
        for label in class_vocab:
            rec[target_column_map[label]] = probs.get(label, 0.0) if is_labeled else 0.0
        if is_labeled:
            unseen_mass = sum(v for k, v in probs.items() if k not in class_vocab)
        rec["target_prob__OTHER_UNSEEN"] = unseen_mass
        rows.append(rec)

    return pd.DataFrame.from_records(rows)


def reshape_cases_to_images(cases_df: pd.DataFrame) -> pd.DataFrame:
    base_cols = [
        c for c in cases_df.columns if c not in IMAGE_PATH_COLUMNS and c not in IMAGE_SHOT_TYPE_COLUMNS
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
    case_image_count = image_df.groupby(GROUP_COLUMN)["image_path"].transform("count")
    image_df["case_image_count"] = case_image_count.astype("int8")
    image_df["sample_weight_case_inverse"] = (1.0 / case_image_count).astype("float32")
    return image_df


def case_level_split(
    image_df: pd.DataFrame,
    *,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    case_targets = image_df[
        [GROUP_COLUMN, TARGET_IS_LABELED_COLUMN, TARGET_HARD_COLUMN]
    ].drop_duplicates()
    targets_per_case = case_targets.groupby(GROUP_COLUMN)[TARGET_HARD_COLUMN].nunique(dropna=True)
    if (targets_per_case > 1).any():
        raise ValueError("A case_id maps to multiple target labels; cannot stratify safely.")

    labeled_case_targets = case_targets[case_targets[TARGET_IS_LABELED_COLUMN] == 1]
    unlabeled_case_ids = case_targets[case_targets[TARGET_IS_LABELED_COLUMN] == 0][GROUP_COLUMN].tolist()
    labeled_case_ids = labeled_case_targets[GROUP_COLUMN].tolist()
    stratify_labels = labeled_case_targets[TARGET_HARD_COLUMN].tolist()

    if not labeled_case_ids:
        raise ValueError("No labeled cases found after parsing weighted_skin_condition_label.")

    try:
        train_labeled_case_ids, test_case_ids = train_test_split(
            labeled_case_ids,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_labels,
        )
    except ValueError:
        train_labeled_case_ids, test_case_ids = train_test_split(
            labeled_case_ids,
            test_size=test_size,
            random_state=random_state,
            stratify=None,
        )

    train_case_ids = list(train_labeled_case_ids) + unlabeled_case_ids
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
            (engineered["other_symptoms_no_relevant_symptoms"] == 1) & (engineered["n_other_symptoms"] > 0)
        ).astype("int8")

    if "fitzpatrick_skin_type" in engineered.columns:
        fst_upper = engineered["fitzpatrick_skin_type"].fillna("").astype(str).str.strip().str.upper()
        engineered["fitzpatrick_skin_type_none_identified_flag"] = (fst_upper == "NONE_IDENTIFIED").astype("int8")
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
        engineered["sex_at_birth_missing_flag"] = (sex_upper == "").astype("int8")

    if "condition_duration" in engineered.columns:
        duration_upper = engineered["condition_duration"].fillna("").astype(str).str.strip().str.upper()
        engineered["condition_duration_missing_flag"] = (duration_upper == "").astype("int8")
        engineered["condition_duration_unknown_flag"] = (duration_upper == "UNKNOWN").astype("int8")
        engineered["condition_duration"] = engineered["condition_duration"].replace("", "MISSING")

    if "related_category" in engineered.columns:
        category_upper = engineered["related_category"].fillna("").astype(str).str.strip().str.upper()
        engineered["related_category_missing_flag"] = (category_upper == "").astype("int8")
        engineered["related_category_none_of_above_flag"] = (category_upper == "NONE_OF_THE_ABOVE").astype("int8")
        engineered["related_category"] = engineered["related_category"].replace("", "MISSING")

    return engineered


def _drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [c for c in LEAKAGE_COLUMNS if c in df.columns]
    return df.drop(columns=drop_cols)


def prepare_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    class_vocab: list[str],
    target_column_map: dict[str, str],
    scale_numeric: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str], list[str]]:
    train_df = _drop_leakage_columns(train_df)
    test_df = _drop_leakage_columns(test_df)

    checkbox_cols = _checkbox_columns(train_df.columns.tolist())
    train_df = _engineer_features(train_df, checkbox_cols)
    test_df = _engineer_features(test_df, checkbox_cols)

    metadata_cols = [c for c in ID_COLUMNS if c in train_df.columns]
    target_train = _build_target_frame(train_df, class_vocab, target_column_map)
    target_test = _build_target_frame(test_df, class_vocab, target_column_map)

    internal_target_cols = [
        TARGET_COLUMN,
        TARGET_PROBS_COLUMN,
        TARGET_IS_LABELED_COLUMN,
        TARGET_HARD_COLUMN,
        TARGET_PARSE_ERROR_COLUMN,
    ]
    train_model = train_df.drop(columns=[c for c in internal_target_cols if c in train_df.columns])
    test_model = test_df.drop(columns=[c for c in internal_target_cols if c in test_df.columns])

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
        train_model[col] = train_model[col].fillna("MISSING").astype(str).str.strip().replace("", "MISSING").str.upper()
        test_model[col] = test_model[col].fillna("MISSING").astype(str).str.strip().replace("", "MISSING").str.upper()

    scaled_numeric_columns: list[str] = []
    if scale_numeric and numeric_cols:
        # Keep binary indicator columns as-is; scale only non-binary numeric features.
        scaled_numeric_columns = [c for c in numeric_cols if train_model[c].nunique(dropna=True) > 2]
        if scaled_numeric_columns:
            scaler = StandardScaler()
            train_model[scaled_numeric_columns] = scaler.fit_transform(train_model[scaled_numeric_columns])
            test_model[scaled_numeric_columns] = scaler.transform(test_model[scaled_numeric_columns])

    if categorical_cols:
        train_cat = pd.get_dummies(train_model[categorical_cols], prefix=categorical_cols, dtype="int8")
        test_cat = pd.get_dummies(test_model[categorical_cols], prefix=categorical_cols, dtype="int8")
        test_cat = test_cat.reindex(columns=train_cat.columns, fill_value=0)
    else:
        train_cat = pd.DataFrame(index=train_model.index)
        test_cat = pd.DataFrame(index=test_model.index)

    train_num = train_model[numeric_cols].reset_index(drop=True)
    test_num = test_model[numeric_cols].reset_index(drop=True)
    x_train = pd.concat([train_num, train_cat.reset_index(drop=True)], axis=1)
    x_test = pd.concat([test_num, test_cat.reset_index(drop=True)], axis=1)

    train_out = pd.concat(
        [train_df[metadata_cols].reset_index(drop=True), target_train.reset_index(drop=True), x_train],
        axis=1,
    )
    test_out = pd.concat(
        [test_df[metadata_cols].reset_index(drop=True), target_test.reset_index(drop=True), x_test],
        axis=1,
    )

    feature_columns = x_train.columns.tolist()
    target_prob_columns = [target_column_map[label] for label in class_vocab] + ["target_prob__OTHER_UNSEEN"]
    return train_out, test_out, feature_columns, target_prob_columns, scaled_numeric_columns


def run_pipeline(
    *,
    cases_path: Path,
    labels_path: Path,
    output_dir: Path,
    test_size: float,
    random_state: int,
    scale_numeric: bool,
) -> None:
    cases_df = _normalize_columns(pd.read_csv(cases_path))
    labels_df = _normalize_columns(pd.read_csv(labels_path))

    image_df = reshape_cases_to_images(cases_df)
    merged_df = image_df.merge(labels_df, on=GROUP_COLUMN, how="inner")
    if TARGET_COLUMN not in merged_df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' was not found after merge.")

    merged_df = _attach_target_metadata(merged_df)
    train_df, test_df = case_level_split(merged_df, test_size=test_size, random_state=random_state)
    class_vocab = _class_vocabulary(train_df)
    target_column_map = _build_target_column_map(class_vocab)
    train_out, test_out, feature_columns, target_prob_columns, scaled_numeric_columns = prepare_features(
        train_df, test_df, class_vocab, target_column_map, scale_numeric
    )

    train_labeled_out = train_out[train_out["is_labeled"] == 1].reset_index(drop=True)
    train_unlabeled_out = train_out[train_out["is_labeled"] == 0].reset_index(drop=True)
    test_labeled_out = test_out[test_out["is_labeled"] == 1].reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    train_labeled_out.to_csv(output_dir / "train_labeled.csv", index=False)
    train_unlabeled_out.to_csv(output_dir / "train_unlabeled.csv", index=False)
    test_labeled_out.to_csv(output_dir / "test_labeled.csv", index=False)

    manifest = {
        "cases_path": str(cases_path),
        "labels_path": str(labels_path),
        "target_column": TARGET_COLUMN,
        "target_prob_columns": target_prob_columns,
        "class_vocabulary": class_vocab,
        "target_column_map_rows": [
            {"label": label, "column": target_column_map[label]} for label in class_vocab
        ],
        "group_column": GROUP_COLUMN,
        "random_state": random_state,
        "test_size": test_size,
        "train_rows_total": int(len(train_out)),
        "train_rows_labeled": int(len(train_labeled_out)),
        "train_rows_unlabeled": int(len(train_unlabeled_out)),
        "test_rows_labeled": int(len(test_labeled_out)),
        "target_parse_errors_total": int(train_out["target_parse_error"].sum() + test_out["target_parse_error"].sum()),
        "test_rows_with_unseen_target_mass": int((test_labeled_out["target_prob__OTHER_UNSEEN"] > 0).sum()),
        "scale_numeric": bool(scale_numeric),
        "scaled_numeric_columns": scaled_numeric_columns,
        "train_cases_total": int(train_out[GROUP_COLUMN].nunique()),
        "train_cases_labeled": int(train_labeled_out[GROUP_COLUMN].nunique()),
        "train_cases_unlabeled": int(train_unlabeled_out[GROUP_COLUMN].nunique()),
        "test_cases_labeled": int(test_labeled_out[GROUP_COLUMN].nunique()),
        "feature_count": int(len(feature_columns)),
        "feature_columns": feature_columns,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Saved: {output_dir / 'train_labeled.csv'}")
    print(f"Saved: {output_dir / 'train_unlabeled.csv'}")
    print(f"Saved: {output_dir / 'test_labeled.csv'}")
    print(f"Saved: {output_dir / 'manifest.json'}")
    print(f"Case leakage check passed: no overlapping '{GROUP_COLUMN}' values between train and test.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare SCIN data into leakage-safe supervised + unsupervised sets.")
    parser.add_argument("--cases-path", type=Path, default=Path("data/raw/dataset/scin_cases.csv"))
    parser.add_argument("--labels-path", type=Path, default=Path("data/raw/dataset/scin_labels.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--scale-numeric",
        dest="scale_numeric",
        action="store_true",
        default=True,
        help="Scale non-binary numeric features using StandardScaler fit on train split.",
    )
    parser.add_argument(
        "--no-scale-numeric",
        dest="scale_numeric",
        action="store_false",
        help="Disable numeric feature scaling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(
        cases_path=args.cases_path,
        labels_path=args.labels_path,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        scale_numeric=args.scale_numeric,
    )


if __name__ == "__main__":
    main()

