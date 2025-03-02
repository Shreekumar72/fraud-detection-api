import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load preprocessed data
X = np.load("X.npy")
y = np.load("y.npy")

# Split Dataset (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost Model
model = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
model.fit(X_train, y_train)

# Evaluate Performance
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Improved Model Accuracy: {accuracy:.2f}")
print("📊 Classification Report:\n", classification_report(y_test, y_pred))

# Save Model
joblib.dump(model, "fraud_detection_model.pkl")
print("✅ Model saved as `fraud_detection_model.pkl`")
