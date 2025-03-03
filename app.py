from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer

app = FastAPI()

# ✅ Load the Model and Vectorizer
model = joblib.load("fraud_detection_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")  # Ensure this file exists

# ✅ Root endpoint to verify API is live (Fix for 405 Method Not Allowed)
@app.api_route("/", methods=["GET", "HEAD"])
def home():
    return {"message": "Fraud Detection API is live!"}

# ✅ Define Request Body
class FraudDetectionRequest(BaseModel):
    description: str

# ✅ Fraud Detection Endpoint
@app.post("/predict/")
async def predict(data: FraudDetectionRequest):
    # Convert text to numerical features
    transformed_text = vectorizer.transform([data.description]).toarray()
    prediction = model.predict(transformed_text)[0]

    # Map prediction to labels
    label_mapping = {0: "genuine", 1: "fraud"}
    return {"type": label_mapping[prediction], "reason": "Automated fraud detection model result"}
