# Stress Anveshan

**Real-time multimodal stress detection from wearable physiological signals (WESAD), with a CNN-LSTM ensemble evaluated under Leave-One-Subject-Out cross-validation and served through a FastAPI service.**

This repository accompanies the paper *"A Real-Time Multimodal Stress Detection Framework Using Wearable Physiological Signals and Ensemble Deep Learning"* (R. Sharma, D. Mumtaz, NIT Srinagar). Every number in the paper can be regenerated from the files committed here.

## Headline results (LOSO, 15 subjects, baseline vs. TSST stress)

| Configuration (CNN-LSTM) | Mean accuracy | Std. | Mean F1 | Worst fold |
|---|---|---|---|---|
| Chest-only | 93.59% | 9.92% | 0.8855 | 66.67% (S2) |
| Wrist-only | 92.06% | 7.07% | 0.8893 | 76.56% (S7) |
| **Multimodal (chest + wrist, probability-averaged)** | **95.34%** | 7.34% | **0.9196** | 71.43% (S2) |

Pooled over all folds the fused model reaches 0.9538 accuracy (601 TN, 12 FP, 33 FN, 329 TP). Inference takes about 51 ms per window on a cloud CPU.

## Reproduce the paper in one command

```bash
pip install -r requirements.txt
python reproduce/paper_results.py --figures
```

This rebuilds Tables I–III, the CNN vs. CNN-LSTM Wilcoxon test, the deployed 9-model ensemble result and Figs. 2–5 from the saved LOSO predictions in `results/`. It then checks **43 reported values against the paper** and exits non-zero if any differ. Output goes to `reproduce/output/`.

## Pipeline

```
WESAD raw .pkl ─► synchronised 60 s windows ─► per-modality features ─► CNN-LSTM per LOSO fold ─► probability averaging
  (chest 700 Hz,     (30 s step, 50% overlap,     (HRV, EDA tonic/phasic,     (input: 1 × F feature row)     (chest_all + wrist_all)
   wrist 4–64 Hz)     majority label)              EMG, ACC, statistical)
```

- **Task:** binary, neutral baseline (0) vs. TSST stress (1). Amusement and meditation windows are excluded. That leaves 975 windows: 613 baseline and 362 stress.
- **Model:** the CNN-LSTM in `notebooks/01_cnn_lstm_loso.ipynb` has two Conv1D blocks (64 and 128 filters), then LSTM(64), then Dense(64), then a sigmoid output. Each window is fed as one engineered feature row of shape `1 × F`, so the convolutional and recurrent layers act as learned non-linear transforms of the features rather than modelling a time sequence (paper, Sec. III-C).
- **Evaluation:** LOSO cross-validation. Median imputation, z-scaling and class weights are fit on the training subjects only in every fold.
- **Fusion:** a chest-only and a wrist-only CNN-LSTM, trained in the same fold, are averaged (Eq. 1).

## Repository layout

```
├── reproduce/
│   └── paper_results.py         ← regenerates every table/figure in the paper
├── results/                     ← saved LOSO outputs (inputs to the script above)
│   ├── dl_cnn_lstm_predictions.csv     one row per test window × experiment
│   ├── dl_cnn_lstm_results_detailed.csv  one row per fold × experiment
│   └── dl_cnn_*.csv                    same for the CNN comparison model
├── notebooks/
│   ├── 01_cnn_lstm_loso.ipynb   ← MAIN MODEL: LOSO training for all 12 sensor configs + production models
│   ├── 02_all_label_robustness.ipynb  how the models react to amusement/meditation (limitation analysis)
│   ├── 03_inference_timing.ipynb      latency numbers in Sec. V-E
│   ├── exploration/             early data inspection and prototypes (not needed to reproduce)
│   └── archive/                 raw-signal LSTM experiment (not used in the paper)
├── scripts/
│   ├── 01_make_windows.py       raw WESAD → synchronised 60 s windows
│   ├── check_windows.py         sanity-check the generated window files
│   └── feature_visualizations.py exploratory feature plots
├── api/                         FastAPI inference service (deployed on Render)
│   ├── feature_extraction/      feature code shared by training and serving
│   ├── model/                   12 trained CNN-LSTM bundles (model, scaler, medians, feature order)
│   └── tests/                   smoke tests (no real data needed)
├── classical_baselines/         RF / SVM / XGBoost preliminary benchmarking + SHAP (not part of reported results)
├── data/README.md               how to obtain WESAD and where to put it
└── render.yaml                  Render deployment config
```

## Full reproduction from raw data

The steps below retrain everything. You need the WESAD dataset (see [`data/README.md`](data/README.md)) and preferably a GPU, such as the free Colab T4.

```bash
pip install -r requirements.txt

# 1. Raw WESAD → synchronised 60 s windows        (writes WDM_dataset/Interim/)
python scripts/01_make_windows.py
python scripts/check_windows.py                    # optional sanity check

# 2. Windows → feature parquets                    (writes WDM_dataset/Features/)
python api/feature_extraction/run_extraction.py
```

Set `WDM_ROOT=/path/to/WDM_dataset` if the data is not in `./WDM_dataset`.

3. Open `notebooks/01_cnn_lstm_loso.ipynb`, point `BASE_PATH` at `WDM_dataset/Features` and run all cells. This writes `dl_cnn_lstm_results_detailed.csv` and `dl_cnn_lstm_predictions.csv`. Copy them into `results/` and rerun `python reproduce/paper_results.py`.

Deep models are sensitive to random initialisation. The paper reports that an independent retrain of the fused CNN-LSTM gave 93.97%, so expect differences of 1–2 percentage points between runs. The committed `results/` files are the exact run reported in the paper.

## Run the API

```bash
cd api
pip install -r requirements.txt
uvicorn main:app --port 8000          # interactive docs at http://localhost:8000/docs
pytest tests/ -v                      # smoke tests
```

See [`api/README.md`](api/README.md) for endpoints and request format. The API averages whichever per-modality models match the sensors in a request (graceful degradation). With all nine available it scores 95.35% LOSO accuracy, which `reproduce/paper_results.py` also checks.

## Known gaps

- **LSTM row of Table II.** The feature-based LSTM notebook and its LOSO outputs are not in this repository yet. The CNN-LSTM and CNN rows are fully reproducible. `notebooks/archive/lstm_raw_signal_exploratory.ipynb` is a separate raw-signal experiment and did **not** produce that row.
- **CNN training notebook.** `results/dl_cnn_*` are committed, but the notebook that produced them is not. It shares the CNN-LSTM notebook's loader, preprocessing and LOSO loop, with a different `build_model`.

## Citation

```bibtex
@inproceedings{sharma_stress_anveshan,
  title  = {A Real-Time Multimodal Stress Detection Framework Using Wearable
            Physiological Signals and Ensemble Deep Learning},
  author = {Sharma, Rajeev and Mumtaz, Deebha},
  note   = {National Institute of Technology, Srinagar}
}
```

WESAD: P. Schmidt et al., *Introducing WESAD, a multimodal dataset for wearable stress and affect detection*, ICMI 2018. WESAD's own licence applies to the dataset, which is not redistributed here.
