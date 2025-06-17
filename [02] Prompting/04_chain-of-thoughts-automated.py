# Chain of Thoughts Prompting using Gemini API (Google GenAI)
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
import json

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
You are an AI assistant who is expert in breaking down the problems and then resolve the user query.

For the given user input, analyze the input and break down the problem step by step.
Atleast think 5-6 steps on how to solve the problem before solving it down.

The steps are: you get a user input, you analyze, you think, you again think for several times and return an output with explaination and then finally you validate the output as well before giving final result.
Follow the steps in sequence as "analyze", "think", "output", "validate" and then finally "result".

Rules: 
1. Follow the strict JSON output as per output schema
2. Always perform one step at a time and wait for next input
3. Carefully analyze user query

Output format: 
{ step: "string", content: "string" }

Example:
Input: What is 2 + 2?
Output: { step: "analyze", content: "Alright! The user is interested in maths query and he's asking a basic arithmatic operation." }
Output: { step: "think", content: "To perform the addition I must go from left to right and add all the operands" }
Output: { step: "output", content: "4" }
Output: { step: "validate", content: "Seems like 4 is correct ans for 2 + 2" }
Output: { step: "result" content: "2 + 2 = 4 and it's calculate by adding all numbers" }
"""

# Empty list to store conversation contents
contents = []

# Initial user input to start the conversation
query = input("Ask Anything > ") # e.g. "What came first egg or chicken?", "What is 4 + 4 - 4 * 4 / 4 + 100?"
contents.append(
    types.Content(
        role="user",
        parts=[
            types.Part.from_text(text=query)
        ]
    )
)

# Automated Chain of Thoughts
# The model will generate a series of steps to solve the problem
while True:
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
        ),
        contents=contents
    )

    parsed_response = json.loads(response.text) # Parse the JSON response 
    contents.append(
        types.Content(
            role="model",
            parts=[
                types.Part.from_text(text=json.dumps(parsed_response)) # Stringify the JSON object
            ]
        )
    )

    if parsed_response.get("step") != "result":
        print(f"🧠: { parsed_response.get("content")}")
        continue

    print(f"🤖: { parsed_response.get("content")}")
    break
