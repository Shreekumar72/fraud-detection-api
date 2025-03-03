from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
import os

app = FastAPI()

# ✅ Load the Model and Vectorizer
model = joblib.load("fraud_detection_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")  # Make sure you save vectorizer too!

# ✅ Define Request Body
class FraudDetectionRequest(BaseModel):
    description: str

@app.get("/")  # Root endpoint to check if API is live
async def root():
    return {"message": "Fraud Detection API is Live!"}

@app.post("/predict/")
async def predict(data: FraudDetectionRequest):
    # Convert text to numerical features
    transformed_text = vectorizer.transform([data.description]).toarray()
    prediction = model.predict(transformed_text)[0]

    # Map prediction to labels
    label_mapping = {0: "genuine", 1: "fraud"}
    return {"type": label_mapping[prediction], "reason": "Automated fraud detection model result"}

# ✅ Ensure Render Uses Correct Port
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))  # Default to 8000 if PORT is not set
    uvicorn.run(app, host="0.0.0.0", port=port)
