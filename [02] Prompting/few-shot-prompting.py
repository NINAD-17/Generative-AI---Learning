# Few Shot Prompting with Gemini API (Google GenAI)
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

# Load Environment Variables
load_dotenv()

# Get API Key from .env file
api_key = os.getenv("GEMINI_API_KEY")

# Initialize Google GenAI Client
client = genai.Client(
    api_key=api_key,
)

# Define the system prompt for the model
system_prompt = """
You are an AI assistant who is specialized in Maths.
You should not answer any query that is not related to Maths.

For a given query help user to solve that along with explaination.

Example:
Input: 2 + 2
Output: 2 + 2 is 4 which is calculated by adding 2 with 2.

Input: 3 * 10
Output: 3 * 10 is 30 which is calculated by multiplying 3 with 10. Funfact: You can even multiply 10 * 3 which will give same answer.

Input: Why is sky blue?
Output: Sorry!, I can only help with Maths questions.
"""

# Generate response using the model
response = client.models.generate_content(
    model="gemini-2.0-flash",
    config=types.GenerateContentConfig(
        system_instruction=system_prompt,
        max_output_tokens=100,
        temperature=0.5 # Balanced creativity and accuracy
    ),
    contents=[
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="""What is science?""")
            ]
        )
    ]
)

# Print the response
print(response.text)