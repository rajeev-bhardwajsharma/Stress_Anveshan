# WESAD Stress Detection API — Quick Start

## 1. Install dependencies

```bash
cd api
pip install -r requirements.txt
```

## 2. Models

The trained CNN-LSTM bundles are already in `model/`, and `config.yaml` points there (`models_dir: ./model`). Each bundle folder contains:

```
model/chest_statistical/
    final_model.keras   trained network (Keras 3.13.2)
    scaler.joblib       StandardScaler fit on all training subjects
    medians.npy         per-feature medians for NaN imputation
    meta.json           exact training feature order + input shape
```

To serve different models, replace these folders or change `models_dir` in `config.yaml`.

## 3. Feature modules

The API imports the feature code from `feature_extraction/` in this folder, the same code used to build the training features. To point it elsewhere, set:

```bash
export WESAD_FEATURE_MODULE_DIR=/path/to/feature_extraction
```

## 4. Start the server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## 5. Run tests

```bash
pytest tests/ -v
```

The endpoint smoke tests mock the model registry. The regression tests in `test_predict_one_matches_training_pipeline` load real bundles from `model/` and check that inference reproduces training preprocessing.

---

## How a prediction is made

For each sensor bundle whose signals are present in the request:

1. Features are extracted from the 60 s window with the same `feature_extraction/` code used for training.
2. They are put in the **exact training column order** from `meta.json`. Names are matched case-insensitively, so the extractor's `chest_ECG_hr_mean` matches the training name `chest_ecg_hr_mean`.
3. Missing or NaN features are filled with the training medians (`medians.npy`).
4. The features are z-scaled with the bundle's `scaler.joblib`, residual NaN/Inf values are set to 0, and the model runs on a `(1, 1, F)` input.

The stress probabilities of all bundles used are then averaged (Eq. 1 in the paper). If a sensor is missing, its bundles are skipped, so the API degrades gracefully instead of failing.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/v1/health | Liveness + model readiness |
| GET | /api/v1/models | List loaded models |
| POST | /api/v1/validate | Validate payload only |
| POST | /api/v1/features | Extract + return features |
| POST | /api/v1/predict | Full inference |

Interactive docs: **http://localhost:8000/docs**

---

## Sample curl — health check

```bash
curl http://localhost:8000/api/v1/health
```

## Sample curl — predict (chest only, abbreviated)

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "subject_id": "S2",
    "timestamp": "2026-06-13T10:30:00Z",
    "window_size_sec": 60,
    "available_sensors": {
      "chest": ["ACC","ECG","EMG","EDA","Temp","Resp"],
      "wrist": []
    },
    "signals": {
      "chest": {
        "ACC":  "<42000x3 array>",
        "ECG":  "<42000 samples>",
        "EMG":  "<42000 samples>",
        "EDA":  "<42000 samples>",
        "Temp": "<42000 samples>",
        "Resp": "<42000 samples>"
      },
      "wrist": {}
    }
  }'
```

## Expected prediction response

```json
{
  "subject_id": "S2",
  "timestamp": "2026-06-13T10:30:00Z",
  "models_used": ["CHEST_ALL"],
  "stress_probability": 0.82,
  "non_stress_probability": 0.18,
  "prediction": "STRESSED",
  "confidence_score": 0.82
}
```

---

## Directory layout

```
api/
├── config.yaml                  ← all paths, thresholds, sensor lists
├── main.py                      ← FastAPI app + lifespan startup
├── requirements.txt
├── app/
│   ├── core/
│   │   ├── settings.py          ← config loader singleton
│   │   └── logging.py
│   ├── schemas/
│   │   └── predict.py           ← all Pydantic models
│   ├── features/
│   │   └── extractor.py         ← calls the feature_extraction/ modules
│   ├── models/
│   │   └── loader.py            ← scans model/, loads .keras + scaler + medians + meta
│   ├── inference/
│   │   ├── validator.py         ← shape/NaN/sensor checks
│   │   └── engine.py            ← eligibility → predict → ensemble
│   └── api/v1/
│       └── routes.py            ← 5 FastAPI endpoints
└── tests/
    └── test_smoke.py
```
