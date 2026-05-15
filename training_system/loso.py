from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import pandas as pd

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

        # metrics
        acc = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds, average="binary")
        recall = recall_score(y_test, preds, average="binary")
        f1 = f1_score(y_test, preds, average="binary")

        print(f"Acc: {acc:.4f}, F1: {f1:.4f}")

        # store
        results.append({
            "subject": test_sub,
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })

    results_df = pd.DataFrame(results)

    print("\n=== FINAL RESULTS ===")
    print(results_df.mean(numeric_only=True))

    return results_df