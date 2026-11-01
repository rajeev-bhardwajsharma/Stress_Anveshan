# Data

The WESAD dataset is **not** included in this repository. Download it from the authors' page:
<https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html>, and accept their licence.

## Expected layout

Unzip it so that the repository root contains:

```
WDM_dataset/
└── WESAD/
    ├── S2/S2.pkl
    ├── S3/S3.pkl
    └── ...            (S2–S17; S12 does not exist)
```

To keep the data elsewhere, set `WDM_ROOT` to the folder that contains `WESAD/`:

```bash
export WDM_ROOT=/data/WDM_dataset
```

## Generated folders

The pipeline creates these next to `WESAD/`. They are git-ignored.

| Folder | Created by | Contents |
|---|---|---|
| `Interim/<S>/<S>_windows.pkl` | `scripts/01_make_windows.py` | synchronised 60 s windows, 30 s step, one dict per window with every sensor at its native rate plus the majority label |
| `Features/<S>/{chest,wrist}/*.parquet` | `api/feature_extraction/run_extraction.py` | per-subject feature tables |
| `Features/{chest,wrist}/all_<device>_<bundle>.parquet` | same script, merge step | all subjects merged. **This is what the training notebook reads.** |

Feature bundles are `acc`, `eda`, `emg` (chest only), `heart` (ECG on the chest, BVP on the wrist) and `statistical`. Each row has the keys `subject, window_idx, start_time, end_time, label` followed by the feature columns.

Labels use WESAD's codes: 1 = baseline, 2 = stress, 3 = amusement, 4 = meditation. The training notebook keeps only 1 and 2 and maps them to 0 and 1.
