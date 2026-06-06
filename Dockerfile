# Use official Python base image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements first (for Docker layer caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pandas \
    numpy \
    scikit-learn \
    xgboost \
    pydantic

# Copy source code and model files
COPY src/ ./src/
COPY data/processed/xgb_model.pkl ./data/processed/
COPY data/processed/scaler.pkl ./data/processed/

# Expose port
EXPOSE 8000

# Start the API
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]