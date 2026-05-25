"""
benchmark.py
------------
Entry point for the full training grid.

Outputs (written to OUTPUT_DIR):
  results_detailed.csv       per-subject metrics for every model * experiment
  predictions_for_curves.csv  per-window y_true + y_prob for ROC / PR curves

Usage:
  python benchmark.py
  WDM_OUTPUT_DIR=/my/path python benchmark.py   # override output directory
"""
import pandas as pd
from data_loader import load_dataset
from loso import run_loso
from model import get_model
from pathlib import Path
import os
import traceback
# Default: a 'results/' folder next to this script.
# Override without touching code: export WDM_OUTPUT_DIR=/wherever/you/want
OUTPUT_DIR = Path(
    os.environ.get("WDM_OUTPUT_DIR", Path(__file__).resolve().parent / "results")
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Experiments ───────────────────────────────────────────────────────────
EXPERIMENTS = {
    "chest_acc":         {"chest": ["acc"]},
    "chest_eda":         {"chest": ["eda"]},
    "chest_emg":         {"chest": ["emg"]},
    "chest_heart":       {"chest": ["heart"]},
    "chest_statistical": {"chest": ["statistical"]},

    "wrist_acc":         {"wrist": ["acc"]},
    "wrist_eda":         {"wrist": ["eda"]},
    "wrist_heart":       {"wrist": ["heart"]},
    "wrist_statistical": {"wrist": ["statistical"]},

    "chest_all": {"chest": ["acc", "eda", "emg", "heart", "statistical"]},
    "wrist_all": {"wrist": ["acc", "eda", "heart", "statistical"]},
}

# ── 2. Models  scalable modifies model.py and add a string here itemize the string
MODELS = ["rf", "svm","xgboost"]

# ── 3. Run 
all_metrics      = []
all_predictions  = []


for model_name in MODELS:
    print(f"\n{'#'*60}")
    print(f"  MODEL: {model_name.upper()}")
    print(f"{'#'*60}")

    for exp_name, config in EXPERIMENTS.items():
        print(f"\n{'='*50}")
        print(f"  EXPERIMENT: {exp_name}")
        print(f"{'='*50}")

        try:
            df = load_dataset(config)
            model = get_model(model_name)
            # run_loso now returns (metrics_df, predictions_df)
            metrics_df, predictions_df = run_loso(
                df,
                model,
                model_name=model_name,
                exp_name=exp_name,
            )

           #  both DataFrames so rows are self-identifying after concat
            metrics_df.insert(0, "experiment", exp_name)
            metrics_df.insert(0, "model",      model_name)

            all_metrics.append(metrics_df)
            all_predictions.append(predictions_df)

        except Exception as e:
            print(f"   Problem {model_name} / {exp_name}: {e}")
            traceback.print_exc()
            continue

if all_metrics:
    detailed_df = pd.concat(all_metrics, ignore_index=True)
    out_metrics = OUTPUT_DIR / "results_detailed.csv"
    detailed_df.to_csv(out_metrics, index=False)
    print(f"\n Saved: {out_metrics}  ({len(detailed_df)} rows)")

    print("\n── Mean F1 by model & experiment ──")
    print(
        detailed_df
        .groupby(["model", "experiment"])["f1"]
        .mean()
        .round(3)
        .to_string()
    )

if all_predictions:
    curves_df = pd.concat(all_predictions, ignore_index=True)
    out_curves = OUTPUT_DIR / "predictions_for_curves.csv"
    curves_df.to_csv(out_curves, index=False)
    print(f"\n Saved: {out_curves}  ({len(curves_df)} rows)")
    print(f"   Columns: {list(curves_df.columns)}")