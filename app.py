from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer

app = FastAPI()

# ✅ Load the Model and Vectorizer
try:
    model = joblib.load("fraud_detection_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")  # Ensure this file exists
except Exception as e:
    raise RuntimeError(f"Error loading model/vectorizer: {str(e)}")

# ✅ Define Request Body
class FraudDetectionRequest(BaseModel):
    description: str

@app.get("/")
async def home():
    """Root endpoint to check if API is running."""
    return {"message": "Fraud Detection API is live!", "docs": "/docs"}

@app.post("/predict/")
async def predict(data: FraudDetectionRequest):
    """Predict whether the app description is fraud or genuine."""
    try:
        # Convert text to numerical features
        transformed_text = vectorizer.transform([data.description]).toarray()
        prediction = model.predict(transformed_text)[0]

        # Map prediction to labels
        label_mapping = {0: "genuine", 1: "fraud"}
        return {"type": label_mapping[prediction], "reason": "Model classified this app as " + label_mapping[prediction]}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
