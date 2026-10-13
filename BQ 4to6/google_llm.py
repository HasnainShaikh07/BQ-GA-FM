from google import genai
import os
from dotenv import load_dotenv


load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


# Paste your API key here, or set it as an environment variable GEMINI_API_KEY
client = genai.Client(api_key=GEMINI_API_KEY)

response = client.models.generate_content(
    model="gemini-3.5-flash",  # free-tier friendly model
    contents="Why sky is dark at night"
)

print(response.text)