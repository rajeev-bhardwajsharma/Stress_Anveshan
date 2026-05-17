"""
loso.py
-------
Leave-One-Subject-Out (LOSO) cross-validation.

Returns
-------
metrics_df     : one row per subject, all performance metrics
predictions_df : one row per window, with y_true + y_prob
                 → used downstream to draw ROC / PR curves
"""

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score,confusion_matrix #added roc_auc and confusion matrix
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np #added numpy

#making non features column global varibale
NON_FEATURE_COLS = ["label", "subject", "window_idx", "start_time", "end_time"]

def run_loso(
        df: pd.DataFrame,
        model,
        model_name:str="unknown",
        exp_name:str="unknown",
) -> tuple[pd.DataFrame,pd.DataFrame]:
    """
    Run LOSO cross-validation.

    Parameters
    
    df         : merged, binarized DataFrame from load_dataset()
    model      : unfitted sklearn-compatible estimator
    model_name : string label, e.g. "rf" — written into predictions_df
    exp_name   : string label, e.g. "chest_all" — written into predictions_df

    Returns
    
    metrics_df     : pd.DataFrame  — per-subject performance metrics
    predictions_df : pd.DataFrame  — per-window true labels + predicted probs
                     columns: model, experiment, subject, window_idx,
                              y_true, y_prob
    """
    
    subjects = df["subject"].unique()
    metrics_rows=[] 
    prediction_rows=[]

    for test_sub in subjects:
        print(f"\n--- Testing on {test_sub} ---")
     
        
        # split
        train_df = df[df["subject"] != test_sub]
        test_df  = df[df["subject"] == test_sub]

        X_train = train_df.drop(columns=NON_FEATURE_COLS)
        
       
        y_train = train_df["label"]

       
        X_test  = test_df.drop(columns=NON_FEATURE_COLS)
        y_test = test_df["label"]
        # Keeping  window_idx so predictions can be traced back to raw windows
        window_idx_test = test_df["window_idx"].values
        
        
         #Imputation (median from train only  no data leakage) 
        # WHY median not mean: robust to outlier spikes common in physiological
        # signals (e.g. EDA artefacts, motion spikes in ACC).
        medians = X_train.median()
        X_train = X_train.fillna(medians)
        X_test  = X_test.fillna(medians)

        print("Train shape:", X_train.shape)
        print("Test shape:", X_test.shape)
        #scaling
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test  = scaler.transform(X_test)#scaling only using  training dataset
        # train
        model.fit(X_train, y_train)

        # predict
        preds = model.predict(X_test)
        
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)[:, 1]
        else:
            # Fallback when the model doesnt provide us probability we just use the scaled confidence values.: use decision_function and min-max normalise to [0,1]
            # so the output is still a usable soft score.
            scores = model.decision_function(X_test)
            probs = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        
            # ── Per-window prediction rows ────────────────────────────────────────
        for idx, (true_label, prob) in enumerate(zip(y_test.values, probs)):
            prediction_rows.append({
                "model":      model_name,
                "experiment": exp_name,
                "subject":    test_sub,
                "window_idx": window_idx_test[idx],
                "y_true":     int(true_label),
                "y_prob":     float(prob),
            })
        try:
            auc = roc_auc_score(y_test, probs)
        except ValueError:
            # Happens if test subject has only one class present
            auc = np.nan
        # metrics
        acc = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds, average="binary", zero_division=0)
        recall = recall_score(y_test, preds, average="binary", zero_division=0)
        f1 = f1_score(y_test, preds, average="binary", zero_division=0)
        
        # Confusion Matrix & Per Class Metrics
        # labels=[0,1] ensures a 2x2 matrix even if test set lacks a class
        cm = confusion_matrix(y_test, preds, labels=[0, 1]) 
        tn, fp, fn, tp = cm.ravel()
        
        # Class 0 (Baseline) specific metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # Recall for Class 0

        print(f"Acc: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

       

        # store
        metrics_rows.append({
            "subject": test_sub,
            "accuracy": acc,
            "precision": precision, # Class 1 (Stress) Precision
            "recall": recall,       # Class 1 (Stress) Recall
            "specificity": specificity, # Class 0 (Baseline) Recall
            "f1": f1,
            "roc_auc": auc,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp
        })

    metrics_df = pd.DataFrame(metrics_rows)
    predictions_df=pd.DataFrame(prediction_rows)
    _print_summary(metrics_df)

    return metrics_df, predictions_df
def _print_summary(metrics_df: pd.DataFrame) -> None:
    # Calculate Global Confusion Matrix
    total_tn = metrics_df['tn'].sum()
    total_fp = metrics_df['fp'].sum()
    total_fn = metrics_df['fn'].sum()
    total_tp = metrics_df['tp'].sum()

    print("\n=== GLOBAL CONFUSION MATRIX ===")
    print(f"True Negatives (Correct Baseline) : {total_tn}")
    print(f"False Positives (False Alarms)    : {total_fp}")
    print(f"False Negatives (Missed Stress)   : {total_fn}")
    print(f"True Positives (Correct Stress)   : {total_tp}")

    print("\n=== FINAL AVERAGED RESULTS ACROSS SUBJECTS ===")
    # Drop confusion matrix raw counts from mean calculation to avoid confusion
    avg_metrics = metrics_df.drop(columns=['subject', 'tn', 'fp', 'fn', 'tp']).mean(numeric_only=True)
    print(avg_metrics.to_string())
