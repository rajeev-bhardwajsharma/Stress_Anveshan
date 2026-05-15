# train.py

from data_loader import load_dataset
from model import get_model
from loso import run_loso
import pandas as pd


def main():

    # 🔹 1. Define what data you want (NO hardcoding inside code)
    config = {
        "chest": ["statistical"]
    }

    # 🔹 2. Load dataset
    df = load_dataset(config)

    # 🔹 3. Convert labels (VERY IMPORTANT)
    # stress = 1, non-stress = 0
    df["label"] = df["label"].apply(lambda x: 1 if x == 2 else 0)

    print("Label distribution after conversion:")
    print(df["label"].value_counts())

    # 🔹 4. Get model
    models = {
    "rf": get_model("rf"),
    "svm": get_model("svm")
    }

    all_results = {}

    for name,model in models.items():
        print(f"current model is {name}")
        results=run_loso(df,model)

        all_results[name]=results

   

    summary = []
    
    for name, res_df in all_results.items():
        avg = res_df.mean(numeric_only=True)
    
        summary.append({
            "model": name,
            "accuracy": avg["accuracy"],
            "precision": avg["precision"],
            "recall": avg["recall"],
            "f1": avg["f1"]
        })
    
    summary_df = pd.DataFrame(summary)
    
    print("\nMODEL COMPARISON ")
    print(summary_df)


if __name__ == "__main__":
    main()