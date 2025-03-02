import numpy as np
import pandas as pd
import json
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load fraud & genuine datasets
with open("fraud-apps.json", "r", encoding="utf-8") as f:
    fraud_data = json.load(f)

with open("genuine-apps.json", "r", encoding="utf-8") as f:
    genuine_data = json.load(f)

# Extract Features
def extract_features(app):
    return {
        "description": app["description"],
        "category": app["category"],
        "developer_name_length": len(app["developer"]["name"]),
        "num_installs": app.get("installs", 0),
        "rating": app.get("rating", 0)
    }

fraud_features = [extract_features(app) for app in fraud_data]
genuine_features = [extract_features(app) for app in genuine_data]

# Convert to DataFrame
df_fraud = pd.DataFrame(fraud_features)
df_genuine = pd.DataFrame(genuine_features)

# Combine both datasets
df = pd.concat([df_fraud, df_genuine], ignore_index=True)
labels = np.concatenate([np.ones(len(df_fraud)), np.zeros(len(df_genuine))])

# **TF-IDF for Description**
vectorizer = TfidfVectorizer(max_features=500)
tfidf_features = vectorizer.fit_transform(df["description"]).toarray()

# **Convert Categorical Data (Category) to One-Hot Encoding**
df = pd.get_dummies(df, columns=["category"])

# **Normalize numerical features**
scaler = StandardScaler()
numeric_features = scaler.fit_transform(df[["developer_name_length", "num_installs", "rating"]])

# **Combine all features**
X = np.hstack((tfidf_features, numeric_features))

# Save vectorizer & scaler
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(scaler, "scaler.pkl")

# Save preprocessed data
np.save("X.npy", X)
np.save("y.npy", labels)

print("✅ Feature Engineering Complete! Saved `X.npy` and `y.npy`")
