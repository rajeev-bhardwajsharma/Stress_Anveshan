"""
train.py-> Debugger file not intended to use in pipeline
--------
Quick single-experiment runner for development and debugging.
Useing benchmark.py for the full grid.

Usage:
  python train.py
"""

from data_loader import load_dataset
from loso import run_loso
from model import get_model
import pandas as pd


def main():
    # ── 1. Define experiment
    config = {"chest": ["statistical"]}

    # ── 2. Load dataset 
    df = load_dataset(config)

    print("Label distribution (0=baseline, 1=stress):")
    print(df["label"].value_counts())

    # ── 3. Run LOSO for each model
    model_names = ["rf", "svm"]
    summary = []

    for name in model_names:
        print(f"\n{'=' * 40}")
        print(f"  Model: {name.upper()}")
        print(f"{'=' * 40}")

        model = get_model(name)
        # run_loso now returns (metrics_df, predictions_df)
        metrics_df, _ = run_loso(df, model, model_name=name, exp_name="chest_statistical")

        avg = metrics_df.mean(numeric_only=True)
        summary.append({
            "model":       name,
            "accuracy":    avg["accuracy"],
            "precision":   avg["precision"],
            "recall":      avg["recall"],
            "specificity": avg["specificity"],
            "f1":          avg["f1"],
            "roc_auc":     avg["roc_auc"],
        })

    #  4. Summary table 
    summary_df = pd.DataFrame(summary)
    print("\n── MODEL COMPARISON ──")
    print(summary_df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()