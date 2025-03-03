from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import uvicorn

# ✅ Initialize FastAPI
app = FastAPI()

# ✅ Load Model & Vectorizer (Ensure Files Exist)
try:
    model = joblib.load("fraud_detection_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
except Exception as e:
    raise RuntimeError(f"Error loading model/vectorizer: {e}")

# ✅ Fix: Root Endpoint to Confirm API is Running
@app.api_route("/", methods=["GET", "HEAD"])
def home():
    return {"message": "Fraud Detection API is live!"}

# ✅ Define Input Schema
class FraudDetectionRequest(BaseModel):
    description: str

# ✅ Fraud Prediction Endpoint (Fix Path)
@app.post("/predict/")
async def predict(data: FraudDetectionRequest):
    try:
        transformed_text = vectorizer.transform([data.description]).toarray()
        prediction = model.predict(transformed_text)[0]

        # Map prediction to labels
        label_mapping = {0: "genuine", 1: "fraud"}
        return {"type": label_mapping[prediction], "reason": "Automated fraud detection model result"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")

# ✅ Fix: Ensure Correct Port Binding
if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))  # Use $PORT from Render
    uvicorn.run(app, host="0.0.0.0", port=port)
