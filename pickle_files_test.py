from pathlib import Path
import pickle

subjects = [f"S{i}" for i in range(2, 18) if i != 12]

for subject in subjects:
    pkl_path = Path("WDM_dataset/Interim") / subject / f"{subject}_windows.pkl"
    
    if not pkl_path.exists():
        print(f"{subject}:  File missing")
        continue
    
    try:
        with open(pkl_path, "rb") as f:
            windows = pickle.load(f)
        
        print(f"{subject}: ✅ OK ({len(windows)} windows)")
    
    except Exception as e:
        print(f"{subject}:  CORRUPT → {e}")