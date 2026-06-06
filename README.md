# Predictive Maintenance — Turbofan Engine RUL Prediction

An end-to-end machine learning system that predicts the **Remaining Useful Life (RUL)** of industrial turbofan engines using sensor telemetry data, with anomaly detection, experiment tracking, and a production REST API deployed via Docker.

---

## Project Overview

| Item | Detail |
|---|---|
| **Dataset** | NASA CMAPSS FD001 — 100 turbofan engines, 20,631 sensor readings |
| **Task** | Regression — predict cycles remaining before engine failure |
| **Model** | XGBoost Regressor |
| **RMSE** | 36.69 cycles |
| **MAE** | 25.94 cycles |
| **R²** | 0.7053 |
| **Anomaly Detection** | Isolation Forest — 30% anomaly rate when RUL < 30 cycles |

---

## Project Structure
---
predictive_maintenance/
├── data/
│   ├── raw/                         # NASA CMAPSS dataset
│   └── processed/                   # Cleaned and featured data
├── notebooks/
│   ├── 01_eda.ipynb                 # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb # 70 engineered features
│   ├── 03_modeling.ipynb            # XGBoost + MLflow tracking
│   └── 04_anomaly_detection.ipynb   # Isolation Forest
├── src/
│   └── api.py                       # FastAPI REST API
├── Dockerfile
├── requirements.txt
└── README.md

## Key Features

- **Time-series feature engineering** — rolling statistics, lag features, rate-of-change, FFT-based frequency features across 14 sensors
- **XGBoost RUL prediction** — R² of 0.70 on validation set
- **Isolation Forest anomaly detection** — detects 30% anomaly rate in final 30 cycles before failure vs <1% when healthy
- **MLflow experiment tracking** — parameters, metrics, and model artifacts logged
- **FastAPI REST API** — real-time RUL prediction endpoint
- **Docker deployment** — fully containerized and portable

---

## Quickstart

### 1. Clone the repository
```bash
git clone https://github.com/Anjali12213443/predictive-maintenance-rul.git
cd predictive-maintenance-rul
```

### 2. Run with Docker
```bash
docker build -t predictive-maintenance-api .
docker run -p 8000:8000 predictive-maintenance-api
```

### 3. Test the API
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
-H "Content-Type: application/json" \
-d '{
  "engine_id": 1,
  "cycle": 180,
  "sensor2": 642.5,
  "sensor3": 1590.0,
  "sensor4": 1410.0,
  "sensor7": 553.0,
  "sensor8": 2388.1,
  "sensor9": 9060.0,
  "sensor11": 47.5,
  "sensor12": 521.0,
  "sensor13": 2388.1,
  "sensor14": 8140.0,
  "sensor15": 8.44,
  "sensor17": 393.0,
  "sensor20": 38.8,
  "sensor21": 23.3
}'
```

### Expected Response
```json
{
  "engine_id": 1,
  "cycle": 180,
  "predicted_rul": 158.74,
  "status": "HEALTHY",
  "message": "Engine operating normally"
}
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| POST | `/predict` | Predict RUL from sensor readings |
| GET | `/docs` | Interactive Swagger UI |

---

## Health Status Logic

| Predicted RUL | Status | Action |
|---|---|---|
| < 30 cycles | CRITICAL | Immediate maintenance required |
| 30–60 cycles | WARNING | Schedule maintenance soon |
| > 60 cycles | HEALTHY | Engine operating normally |

---

## Tech Stack

- **Python** — Pandas, NumPy, Scikit-learn
- **ML** — XGBoost, Isolation Forest
- **Tracking** — MLflow
- **API** — FastAPI, Uvicorn
- **Deployment** — Docker
- **Dataset** — NASA CMAPSS FD001
