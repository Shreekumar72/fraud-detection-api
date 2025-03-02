import requests

# API Key from RapidAPI
API_KEY = "a80186586dmshc4fd65a68008663p156bd7jsnf2e484111d07"

# API Endpoint
url = "https://app-store-and-google-play-api.p.rapidapi.com/v1/google-play/search"

# Parameters - FIX: Add the "text" field with a sample search query
params = {
    "text": "WhatsApp",  # Search query (Required field)
    "country": "us",  # Country Code
    "language": "en"   # Language
}

# Headers for Authentication
headers = {
    "x-rapidapi-host": "app-store-and-google-play-api.p.rapidapi.com",
    "x-rapidapi-key": API_KEY
}

# Send Request
response = requests.get(url, headers=headers, params=params)

# Print Response
if response.status_code == 200:
    print("Response Data:", response.json())
else:
    print("Error:", response.status_code, response.text)


