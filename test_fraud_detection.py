import json
from analyze_fraud_gemini import analyze_fraud

# Load labeled datasets
with open("fraud-apps.json", "r", encoding="utf-8") as f:
    fraud_apps = json.load(f)


with open("genuine-apps.json", "r", encoding="utf-8") as f:
    genuine_apps = json.load(f)


# Initialize counters
correct_fraud = 0
correct_genuine = 0
suspected_cases = 0

# Process Fraudulent Apps
print("\n🔴 Testing Fraudulent Apps...\n")
for app in fraud_apps:
    result = analyze_fraud(app)
    print(f"App: {app['title']} → Prediction: {result['type']} | Reason: {result['reason']}")
    
    if result["type"] == "fraud":
        correct_fraud += 1
    elif result["type"] == "suspected":
        suspected_cases += 1

# Process Genuine Apps
print("\n🟢 Testing Genuine Apps...\n")
for app in genuine_apps:
    result = analyze_fraud(app)
    print(f"App: {app['title']} → Prediction: {result['type']} | Reason: {result['reason']}")
    
    if result["type"] == "genuine":
        correct_genuine += 1
    elif result["type"] == "suspected":
        suspected_cases += 1

# Generate Report
total_fraud = len(fraud_apps)
total_genuine = len(genuine_apps)

print("\n📊 **Model Performance Report**")
print(f"✅ Correctly Detected Fraud Apps: {correct_fraud}/{total_fraud} ({(correct_fraud/total_fraud)*100:.2f}%)")
print(f"✅ Correctly Detected Genuine Apps: {correct_genuine}/{total_genuine} ({(correct_genuine/total_genuine)*100:.2f}%)")
print(f"⚠️ Suspected Cases: {suspected_cases}")

