from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scin_data_modeling.models.semi_supervised import run_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SCIN semi-supervised models and print per-model results.")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/semi_supervised"))
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--self-training-threshold", type=float, default=0.6)
    parser.add_argument("--pseudo-threshold", type=float, default=0.6)
    parser.add_argument("--pseudo-margin", type=float, default=0.1)
    parser.add_argument("--top-k-classes", type=int, default=15)
    parser.add_argument("--min-label-confidence", type=float, default=0.55)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_benchmark(
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
        random_state=args.random_state,
        self_training_threshold=args.self_training_threshold,
        pseudo_threshold=args.pseudo_threshold,
        pseudo_margin=args.pseudo_margin,
        top_k_classes=args.top_k_classes,
        min_label_confidence=args.min_label_confidence,
    )

    metrics_path = args.output_dir / "semi_supervised_metrics.csv"
    summary_path = args.output_dir / "semi_supervised_summary.json"
    pseudo_path = args.output_dir / "unlabeled_pseudo_labels_accepted.csv"

    metrics_df = pd.read_csv(metrics_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    accepted_df = pd.read_csv(pseudo_path)

    print("\n=== Semi-Supervised Model Results ===")
    print(metrics_df.to_string(index=False))
    print("\n=== Run Summary ===")
    print(f"Best method: {summary['best_method']}")
    print(f"Grouped classes used: {summary['n_grouped_classes']} (top_k={summary['top_k_classes']})")
    print(f"Min label confidence filter: {summary['min_label_confidence']:.2f}")
    print(
        "Test rows used: "
        f"{summary['n_test_labeled_rows_seen_classes']} / {summary['n_test_labeled_rows_total']} "
        f"(excluded unseen classes: {summary['n_test_labeled_rows_unseen_classes']})"
    )
    print(f"Pseudo-labels accepted: {len(accepted_df)} / {summary['n_unlabeled_train_rows']}")
    print(f"Pseudo acceptance rate: {summary['pseudo_accept_rate']:.4f}")
    print(
        "Pseudo accepted cases: "
        f"{summary['n_pseudo_accepted_cases']} / {summary['n_unlabeled_train_cases']} "
        f"({summary['pseudo_accept_rate_cases']:.4f})"
    )
    print("\nArtifacts:")
    print(f"- {metrics_path}")
    print(f"- {summary_path}")
    print(f"- {pseudo_path}")


if __name__ == "__main__":
    main()
