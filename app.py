from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer

app = FastAPI()

# ✅ Root endpoint (Supports both GET and HEAD requests)
@app.api_route("/", methods=["GET", "HEAD"])
def home():
    return {"message": "Fraud Detection API is live!"}

# ✅ Load Model and Vectorizer
try:
    model = joblib.load("fraud_detection_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")  # Ensure this file exists
except Exception as e:
    raise RuntimeError(f"Error loading model or vectorizer: {e}")

# ✅ Define Request Body
class FraudDetectionRequest(BaseModel):
    description: str

# ✅ Fraud Detection Endpoint
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
        return {"error": str(e)}
