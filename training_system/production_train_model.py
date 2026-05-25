"""
production_train_model.py
--------------------------
Trains one final model per (model_name × experiment) on the ENTIRE dataset
— no held-out subjects, no LOSO splits.

This is the "production" step: once your LOSO benchmarks (benchmark.py) have
told you which model/experiment combo performs best, run this script once to
produce deployment-ready artefacts.  The saved files never need to be
regenerated unless you retrain on fresh data.

Saved artefacts layout
-----------------------
  saved_production_models/
    <model_name>/
      <exp_name>/
        model.joblib      ← fitted estimator  (trained on 100 % of data)
        scaler.joblib     ← fitted StandardScaler
        medians.joblib    ← per-feature medians used for NaN imputation
        meta.json         ← feature names, label map, training stats

Usage

After training, load a model in any other script with:
  from production_train_model import load_production_model
  model, scaler, medians, meta = load_production_model("rf", "chest_all")
  # then call predict_stress(X_raw_df, model, scaler, medians, meta)
"""

import json
import os
import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


from data_loader import load_dataset
from model import get_model


NON_FEATURE_COLS = ["label", "subject", "window_idx", "start_time", "end_time"]

PROD_MODEL_DIR = Path(
    os.environ.get("WDM_PROD_MODEL_DIR",
                   Path(__file__).resolve().parent / "saved_production_models")
)

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

MODELS = ["rf", "svm", "xgboost"]


def train_and_save_production_model(
    df: pd.DataFrame,
    model_name: str,
    exp_name: str,
) -> Path:
    """
    Train one model on the full dataset (all subjects) and persist it.

    Parameters
    ----------
    df         : merged, binarized DataFrame from load_dataset()
    model_name : e.g. "rf", "svm", "xgboost"
    exp_name   : e.g. "chest_all"

    Returns
    -------
    Path : directory where artefacts were saved
    """
    save_dir = PROD_MODEL_DIR / model_name / exp_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Feature / label split ─────────────────────────────────────────────
    X = df.drop(columns=NON_FEATURE_COLS)
    y = df["label"]

    feature_names = X.columns.tolist()
    n_subjects    = df["subject"].nunique()
    n_windows     = len(df)
    label_counts  = y.value_counts().to_dict()

    print(f"   Subjects : {n_subjects}")
    print(f"   Windows  : {n_windows}")
    print(f"   Features : {len(feature_names)}")
    print(f"   Labels   : {label_counts}  (0=baseline, 1=stress)")

    # ── Imputation — median over the full dataset ─────────────────────────
    # In LOSO we used train-only medians to avoid leakage.
    # Here there is no test split, so full-dataset medians are correct.
    medians = X.median()
    X = X.fillna(medians)

    # ── Scaling ───────────────────────────────────────────────────────────
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Train ─────────────────────────────────────────────────────────────
    model = get_model(model_name)
    model.fit(X_scaled, y)

    # ── Persist ───────────────────────────────────────────────────────────
    joblib.dump(model,   save_dir / "model.joblib")
    joblib.dump(scaler,  save_dir / "scaler.joblib")
    joblib.dump(medians, save_dir / "medians.joblib")

    meta = {
        "model_name":    model_name,
        "experiment":    exp_name,
        "feature_names": feature_names,
        "n_features":    len(feature_names),
        "n_subjects":    n_subjects,
        "n_windows":     n_windows,
        "label_map":     {"0": "baseline", "1": "stress"},
        "label_counts":  {str(k): int(v) for k, v in label_counts.items()},
    }
    with open(save_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"   Saved → {save_dir}")
    return save_dir


# Loading helper  

def load_production_model(model_name: str, exp_name: str):
    """
    Load a previously saved production model.

    Returns
    -------
    model   : fitted sklearn estimator
    scaler  : fitted StandardScaler
    medians : pd.Series  (feature → median value, for NaN imputation)
    meta    : dict       (feature names, label map, training stats)

    Example
    -------
    model, scaler, medians, meta = load_production_model("rf", "chest_all")
    predictions = predict_stress(new_feature_df, model, scaler, medians, meta)
    """
    save_dir = PROD_MODEL_DIR / model_name / exp_name

    missing = [
        f for f in ["model.joblib", "scaler.joblib", "medians.joblib", "meta.json"]
        if not (save_dir / f).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Artefacts not found in {save_dir}: {missing}\n"
            f"Run production_train_model.py first."
        )

    model   = joblib.load(save_dir / "model.joblib")
    scaler  = joblib.load(save_dir / "scaler.joblib")
    medians = joblib.load(save_dir / "medians.joblib")
    with open(save_dir / "meta.json") as f:
        meta = json.load(f)

    return model, scaler, medians, meta



# Inference helper 


def predict_stress(
    X_raw: pd.DataFrame,
    model,
    scaler,
    medians: pd.Series,
    meta: dict,
) -> pd.DataFrame:
    """
    Run inference on raw (unscaled, possibly NaN-containing) feature data.

    Parameters
    ----------
    X_raw   : DataFrame whose columns are the feature names stored in meta.
              It must NOT contain label/subject/window columns.
    model   : fitted estimator from load_production_model()
    scaler  : fitted StandardScaler from load_production_model()
    medians : imputation medians from load_production_model()
    meta    : metadata dict from load_production_model()

    Returns
    -------
    pd.DataFrame with columns:
        predicted_label  — 0 (baseline) or 1 (stress)
        predicted_class  — "baseline" or "stress"
        prob_baseline    — probability of class 0
        prob_stress      — probability of class 1
    """
    expected = meta["feature_names"]

    # Guard: check all expected features are present
    missing_cols = [c for c in expected if c not in X_raw.columns]
    if missing_cols:
        raise ValueError(
            f"Input is missing {len(missing_cols)} feature(s): {missing_cols[:10]}..."
        )

    X = X_raw[expected].copy()         # enforce exact column order
    X = X.fillna(medians)              # impute with training medians
    X_scaled = scaler.transform(X)     # scale with training scaler

    preds = model.predict(X_scaled)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_scaled)   # shape (n, 2)
        prob_baseline = probs[:, 0]
        prob_stress   = probs[:, 1]
    else:
        scores        = model.decision_function(X_scaled)
        prob_stress   = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        prob_baseline = 1.0 - prob_stress

    label_map = meta["label_map"]   # {"0": "baseline", "1": "stress"}
    return pd.DataFrame({
        "predicted_label": preds,
        "predicted_class": [label_map[str(p)] for p in preds],
        "prob_baseline":   prob_baseline,
        "prob_stress":     prob_stress,
    })



# Main — run the full training grid


if __name__ == "__main__":
    summary_rows = []

    for model_name in MODELS:
        print(f"\n{'#'*60}")
        print(f"  MODEL: {model_name.upper()}")
        print(f"{'#'*60}")

        for exp_name, config in EXPERIMENTS.items():
            print(f"\n{'='*50}")
            print(f"  EXPERIMENT: {exp_name}")
            print(f"{'='*50}")

            try:
                df       = load_dataset(config)              # from data_loader.py
                save_dir = train_and_save_production_model(df, model_name, exp_name)

                summary_rows.append({
                    "model":      model_name,
                    "experiment": exp_name,
                    "n_windows":  len(df),
                    "n_subjects": df["subject"].nunique(),
                    "save_dir":   str(save_dir),
                    "status":     "OK",
                })

            except Exception as e:
                print(f"   FAILED {model_name}/{exp_name}: {e}")
                traceback.print_exc()
                summary_rows.append({
                    "model":      model_name,
                    "experiment": exp_name,
                    "status":     f"FAILED: {e}",
                })

    # ── Print summary ─────────────────────────────────────────────────────
    print(f"\n{'#'*60}")
    print("  TRAINING COMPLETE — SUMMARY")
    print(f"{'#'*60}")
    summary_df = pd.DataFrame(summary_rows)
    ok    = summary_df[summary_df["status"] == "OK"]
    fails = summary_df[summary_df["status"] != "OK"]

    print(f"\n  Trained successfully : {len(ok)} / {len(summary_df)}")
    if not fails.empty:
        print("  Failed:")
        for _, row in fails.iterrows():
            print(f"    {row['model']} / {row['experiment']} → {row['status']}")

    print(f"\n  All artefacts saved under: {PROD_MODEL_DIR}")

'''
   print("""
  To use a saved model in another script:

    from production_train_model import load_production_model, predict_stress

    model, scaler, medians, meta = load_production_model("rf", "chest_all")
    results = predict_stress(new_feature_df, model, scaler, medians, meta)
    print(results.head())
""")
'''
