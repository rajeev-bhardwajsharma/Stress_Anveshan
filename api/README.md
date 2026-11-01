# WESAD Stress Detection API — Quick Start

## 1. Install dependencies

```bash
cd wesad_api
pip install -r requirements.txt
```

## 2. Point config to your models

Edit `config.yaml`:
```yaml
paths:
  models_dir: /home/rs/ml-projects/WDM_dataset/Models
model:
  active_tag: cnn_lstm          # ← your MODEL_TAG folder name
  model_filename: final_model.keras
  scaler_filename: scaler.joblib
```

Each bundle directory must contain:
```
Models/cnn_lstm/chest_statistical/
    final_model.keras
    scaler.joblib
```

## 3. Set feature module path (if different from default)

```bash
export WESAD_FEATURE_MODULE_DIR=/home/rs/ml-projects/WDM_dataset
```

## 4. Start the server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## 5. Run smoke tests (no real models needed)

```bash
pytest tests/test_smoke.py -v
```

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
wesad_api/
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
│   │   └── extractor.py         ← calls your 5 feature modules
│   ├── models/
│   │   └── loader.py            ← scans Models/, loads .keras + scaler.joblib
│   ├── inference/
│   │   ├── validator.py         ← shape/NaN/sensor checks
│   │   └── engine.py            ← eligibility → predict → ensemble
│   └── api/v1/
│       └── routes.py            ← 5 FastAPI endpoints
└── tests/
    └── test_smoke.py
```
