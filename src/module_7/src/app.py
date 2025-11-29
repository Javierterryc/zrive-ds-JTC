from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from typing import Any
import os
from basket_model.feature_store import FeatureStore
from basket_model.basket_model import BasketModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST, write_to_textfile, REGISTRY
from fastapi import Response

# Create an instance of FastAPI
app = FastAPI()

feature_store = FeatureStore()
basket_model = BasketModel()

REQUEST_COUNT = Counter(
    "predict_requests_total",
    "Total number of prediction requests"
)

PREDICTION_ERRORS = Counter(
    "prediction_errors_total",
    "Total number of prediction errors"
)

PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Time spent running model prediction"
)

class PredictionRequest(BaseModel):
    user_id: str

class PredictionResponse(BaseModel):
    user_id: str
    prediction: float

METRICS_DIR = "metrics"
METRICS_FILE_PATH = os.path.join(METRICS_DIR, "prometheus_metrics.txt")

def dump_metrics_to_txt():
    os.makedirs(METRICS_DIR, exist_ok=True)  
    write_to_textfile(METRICS_FILE_PATH, REGISTRY)

# Root endpoint 
@app.get("/")
def home():
    return {"status": "API is running"}

# Endpoint for status
@app.get("/status")
def get_status():
    return {"status": "ok"}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):

    REQUEST_COUNT.inc()

    try:
        features = feature_store.get_features(request.user_id)
        with PREDICTION_LATENCY.time(): # Con el with, nos aseguramos que 
            pred = basket_model.predict(features)
        dump_metrics_to_txt()
        return PredictionResponse(
            user_id=request.user_id,
            prediction=pred.mean()
        )
    except Exception as e:
        PREDICTION_ERRORS.inc()
        dump_metrics_to_txt()
        raise e


# Run the server when executing this file directly
# if __name__ == "__main__":
#     uvicorn.run(
#         "app:app",      # module_name:variable_name
#         host="0.0.0.0",
#         port=8002
#     )

# User id example
# 05e0a7ebc5f1e0474ae3f3a2d06443dd5a6f44c677ef13e2aab410b2435030fd9cf25b4da999f892614136566cf6845ac243b8bec7957cb4d0040c463bf7578e


# How to run this app: execute this in the terminal
# uvicorn app:app --host 0.0.0.0 --port 8002