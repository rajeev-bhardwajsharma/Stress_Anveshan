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

def run_loso(df, model):
    subjects = df["subject"].unique()

    results = []

    for test_sub in subjects:
        print(f"\n--- Testing on {test_sub} ---")
     
        
        # split
        train_df = df[df["subject"] != test_sub]
        test_df  = df[df["subject"] == test_sub]

        # features / labels
        NON_FEATURE_COLS = ["label", "subject", "window_idx", "start_time", "end_time"]
        X_train = train_df.drop(columns=NON_FEATURE_COLS)
        
       
        y_train = train_df["label"]

       
        X_test  = test_df.drop(columns=NON_FEATURE_COLS)
        y_test = test_df["label"]
        
        #for imputation
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
         # ROC AUC Requires Probabilities for Class 1
        try:
            probs = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, probs)
        except ValueError:
            # Handles edge case where test subject might only have 1 class entirely
            auc = np.nan
        except AttributeError:
            # Fallback if a model without predict_proba is used
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
        results.append({
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

    results_df = pd.DataFrame(results)
    # Calculate Global Confusion Matrix
    total_tn = results_df['tn'].sum()
    total_fp = results_df['fp'].sum()
    total_fn = results_df['fn'].sum()
    total_tp = results_df['tp'].sum()

    print("\n=== GLOBAL CONFUSION MATRIX ===")
    print(f"True Negatives (Correct Baseline) : {total_tn}")
    print(f"False Positives (False Alarms)    : {total_fp}")
    print(f"False Negatives (Missed Stress)   : {total_fn}")
    print(f"True Positives (Correct Stress)   : {total_tp}")

    print("\n=== FINAL AVERAGED RESULTS ACROSS SUBJECTS ===")
    # Drop confusion matrix raw counts from mean calculation to avoid confusion
    avg_metrics = results_df.drop(columns=['subject', 'tn', 'fp', 'fn', 'tp']).mean(numeric_only=True)
    print(avg_metrics.to_string())

    return results_df