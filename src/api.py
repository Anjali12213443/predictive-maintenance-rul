import os
import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Dynamic path resolution — works both locally and in Docker
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load model and scaler
with open(os.path.join(BASE_DIR, 'data/processed/xgb_model.pkl'), 'rb') as f:
    model = pickle.load(f)

with open(os.path.join(BASE_DIR, 'data/processed/scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)

# Sensor columns
SENSOR_COLS = [
    'sensor2', 'sensor3', 'sensor4', 'sensor7', 'sensor8',
    'sensor9', 'sensor11', 'sensor12', 'sensor13', 'sensor14',
    'sensor15', 'sensor17', 'sensor20', 'sensor21'
]

# Build exact 70-feature column order matching training
FEATURE_COLS = SENSOR_COLS.copy()
for s in SENSOR_COLS:
    FEATURE_COLS.append(f'{s}_roll_mean')
    FEATURE_COLS.append(f'{s}_roll_std')
    FEATURE_COLS.append(f'{s}_lag1')
    FEATURE_COLS.append(f'{s}_roc')

# FastAPI app
app = FastAPI(
    title="Predictive Maintenance API",
    description="Predicts Remaining Useful Life (RUL) of turbofan engines",
    version="1.0.0"
)

# Input schema
class SensorReading(BaseModel):
    engine_id: int
    cycle: int
    sensor2: float
    sensor3: float
    sensor4: float
    sensor7: float
    sensor8: float
    sensor9: float
    sensor11: float
    sensor12: float
    sensor13: float
    sensor14: float
    sensor15: float
    sensor17: float
    sensor20: float
    sensor21: float

# Health check
@app.get("/")
def root():
    return {"status": "Predictive Maintenance API is running"}

# Prediction endpoint
@app.post("/predict")
def predict_rul(reading: SensorReading):

    # Raw sensor values
    raw = {
        'sensor2': reading.sensor2,
        'sensor3': reading.sensor3,
        'sensor4': reading.sensor4,
        'sensor7': reading.sensor7,
        'sensor8': reading.sensor8,
        'sensor9': reading.sensor9,
        'sensor11': reading.sensor11,
        'sensor12': reading.sensor12,
        'sensor13': reading.sensor13,
        'sensor14': reading.sensor14,
        'sensor15': reading.sensor15,
        'sensor17': reading.sensor17,
        'sensor20': reading.sensor20,
        'sensor21': reading.sensor21
    }

    # Build all 70 features in exact training order
    feature_dict = {}

    # First 14 — raw sensor values
    for s in SENSOR_COLS:
        feature_dict[s] = raw[s]

    # Next 56 — engineered features (4 per sensor)
    for s in SENSOR_COLS:
        feature_dict[f'{s}_roll_mean'] = raw[s]
        feature_dict[f'{s}_roll_std'] = 0.0
        feature_dict[f'{s}_lag1'] = raw[s]
        feature_dict[f'{s}_roc'] = 0.0

    # Create dataframe with exact column order
    feature_df = pd.DataFrame([feature_dict])[FEATURE_COLS]

    # Scale and predict
    feature_scaled = scaler.transform(feature_df)
    rul_prediction = max(0, float(model.predict(feature_scaled)[0]))

    # Health status
    if rul_prediction < 30:
        status = "CRITICAL"
        message = "Immediate maintenance required"
    elif rul_prediction < 60:
        status = "WARNING"
        message = "Schedule maintenance soon"
    else:
        status = "HEALTHY"
        message = "Engine operating normally"

    return {
        "engine_id": reading.engine_id,
        "cycle": reading.cycle,
        "predicted_rul": round(rul_prediction, 2),
        "status": status,
        "message": message
    }