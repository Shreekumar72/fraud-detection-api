from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
import os

# ✅ Create FastAPI app
app = FastAPI()

# ✅ Load the Model and Vectorizer
model = joblib.load("fraud_detection_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")  # Ensure this file exists

# ✅ Define Root Endpoint to Ensure API is Reachable
@app.api_route("/", methods=["GET", "HEAD"])
def home():
    return {"message": "Fraud Detection API is live!"}

# ✅ Define Request Body
class FraudDetectionRequest(BaseModel):
    description: str

# ✅ Fraud Detection Prediction Endpoint
@app.post("/predict/")
async def predict(data: FraudDetectionRequest):
    # Convert text to numerical features
    transformed_text = vectorizer.transform([data.description]).toarray()
    prediction = model.predict(transformed_text)[0]

    # Map prediction to labels
    label_mapping = {0: "genuine", 1: "fraud"}
    return {"type": label_mapping[prediction], "reason": "Automated fraud detection model result"}

# ✅ Start the Uvicorn Server Correctly
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 10000))  # Render assigns PORT dynamically
    uvicorn.run(app, host="0.0.0.0", port=port)
