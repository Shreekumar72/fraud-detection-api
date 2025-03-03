from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import uvicorn

# ✅ Initialize FastAPI App
app = FastAPI()

# ✅ Load the Model and Vectorizer Safely
try:
    model = joblib.load("fraud_detection_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")  # Ensure this file exists
except Exception as e:
    raise RuntimeError(f"Error loading model/vectorizer: {e}")

# ✅ Root Endpoint (Fix: Support HEAD requests)
@app.api_route("/", methods=["GET", "HEAD"])
def home():
    return {"message": "Fraud Detection API is live!"}

# ✅ Define Request Body for Fraud Detection
class FraudDetectionRequest(BaseModel):
    description: str

# ✅ Fraud Detection Prediction Endpoint
@app.post("/predict/")
async def predict(data: FraudDetectionRequest):
    try:
        # Convert text to numerical features
        transformed_text = vectorizer.transform([data.description]).toarray()
        prediction = model.predict(transformed_text)[0]

        # Map prediction to labels
        label_mapping = {0: "genuine", 1: "fraud"}
        return {"type": label_mapping[prediction], "reason": "Automated fraud detection model result"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")

# ✅ Ensure Correct Port Binding (Fix for Render Deployment)
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Render provides $PORT, default 8000 for local
    uvicorn.run(app, host="0.0.0.0", port=port)
