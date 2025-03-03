from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer

app = FastAPI(
    title="Fraud Detection API",
    description="An API to detect fraudulent apps using machine learning.",
    version="1.0",
    docs_url="/docs",  # Ensure docs are enabled
    redoc_url="/redoc"
)

# ✅ Root Endpoint for API health check
@app.get("/")
async def root():
    return {"message": "Fraud Detection API is live!", "docs": "/docs"}

# ✅ Load Model & Vectorizer
try:
    model = joblib.load("fraud_detection_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")  # Ensure this file exists
except Exception as e:
    print(f"Error loading model/vectorizer: {e}")

# ✅ Request Schema
class FraudDetectionRequest(BaseModel):
    description: str

# ✅ Prediction Endpoint
@app.post("/predict/")
async def predict(data: FraudDetectionRequest):
    try:
        transformed_text = vectorizer.transform([data.description]).toarray()
        prediction = model.predict(transformed_text)[0]
        label_mapping = {0: "genuine", 1: "fraud"}
        return {"type": label_mapping[prediction], "reason": "Automated fraud detection model result"}
    except Exception as e:
        return {"error": str(e)}
