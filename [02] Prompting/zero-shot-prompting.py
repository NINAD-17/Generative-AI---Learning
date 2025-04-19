# Using GEMINI with OpenAI Library
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load Environment Variables
load_dotenv()

# Get API Key from .env file
api_key = os.getenv("GEMINI_API_KEY")

# Initialize OpenAI Client
# Note: The base_url is set to the Google Gemini API endpoint for OpenAI.
client = OpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Result from the API
result = client.chat.completions.create(
    model="gemini-2.0-flash",
    messages=[
        {"role": "user", "content": "What is the square root of 16?"},
    ]
)

# Print the result
print(result.choices[0].message.content)