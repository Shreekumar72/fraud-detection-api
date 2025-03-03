from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer

app = FastAPI()

# ✅ Load the Model and Vectorizer
model = joblib.load("fraud_detection_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")  # Make sure this exists

# ✅ Root endpoint to check if API is running
@app.get("/")
def home():
    return {"message": "Fraud Detection API is live!"}

# ✅ Define Request Body
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

