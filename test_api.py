import requests

url = "https://fraud-detection-api-rijr.onrender.com/predict/"
data = {"description": "This app offers high-interest loans"}

response = requests.post(url, json=data)

print("Status Code:", response.status_code)
print("Response:", response.json())
