import pandas as pd
from data_loader import load_dataset
from loso import run_loso
from model import get_model
import traceback

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

# ── 2. Models — scalable modifies model.py and add a string here itemize the string
MODELS = ["rf", "svm"]

# ── 3. Run 
all_results = []

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
            results_df = run_loso(df, model)

            results_df.insert(0, "experiment", exp_name)
            results_df.insert(0, "model", model_name)

            all_results.append(results_df)

        except Exception as e:
            print(f"   Problem {model_name} / {exp_name}: {e}")
            traceback.print_exc()
            continue

# ── 4. Save 
detailed_df = pd.concat(all_results, ignore_index=True)
detailed_df.to_csv("/home/rs/ml-projects/result_of_training/results_detailed.csv", index=False)

print("\n Saved Succefull results_detailed.csv")
print(f"   Total rows: {len(detailed_df)}")
print("\n─ Mean F1 by model & experiment ──")
print(
    detailed_df
    .groupby(["model", "experiment"])["f1"]
    .mean()
    .round(3)
    .to_string()
)