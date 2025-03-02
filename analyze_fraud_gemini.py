import google.generativeai as genai
import json

# Configure the Gemini model
genai.configure(api_key="AIzaSyDbEjxxRv4Xd1lRhU6TZVrsjyhqPLug_tI")  # Replace with your API Key
model = genai.GenerativeModel("gemini-1.5-pro")  # Use an available Gemini model

def analyze_fraud(app_data):
    """
    Analyze the fraud likelihood of an app using Gemini API.
    Returns a structured JSON response.
    """

    # Define a structured prompt to force JSON output
    prompt = f"""
    You are an AI fraud detection assistant. Based on the given app details, classify it as:
    - "fraud": if suspicious activity is detected
    - "genuine": if the app follows standard security practices
    - "suspected": if there are some red flags but not confirmed fraud

    Return **ONLY** a valid JSON object in this exact format:
    {{
        "type": "fraud" / "genuine" / "suspected",
        "reason": "Explain the classification in max 300 characters."
    }}

    Here is the app data:
    {json.dumps(app_data)}
    """

    try:
        response = model.generate_content(prompt)
        response_text = response.text.strip()

        # Extract JSON part only from the response
        start_idx = response_text.find("{")
        end_idx = response_text.rfind("}")
        structured_output = json.loads(response_text[start_idx:end_idx+1])

        return structured_output

    except Exception as e:
        return {"type": "suspected", "reason": f"Parsing error: {str(e)}"}

# Example usage
if __name__ == "__main__":
    # Replace with actual app data fetched from the API
    app_data = {
        "title": "WhatsApp Messenger",
        "developer": "WhatsApp LLC",
        "permissions": ["camera", "microphone", "storage"],
        "user_reviews": ["Excellent app!", "Privacy concerns detected", "Works well"]
    }

    result = analyze_fraud(app_data)
    print("Fraud Analysis Result:", result)
