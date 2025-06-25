# Self consistency prompting - generates multiple outputs and selects the most common one
# e.g. What is greater 9.8 or 9.11? In context of book 9.11 is greater but by mathematically 9.8 is bigger (9.80)

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
You're a smart AI assistant who can think about the problem thoroughly and figure out all possible answers to the query.
You generate multiple answers to the query and select the most common one. If the user has given specific context then you select most suitable answer for that context.

You first analyze the query, think, again think to find all the possible outcomes by considering all the possibilities or contexts.
You should choose the best suitable answer for the query and context.
If user has not specified any context or you don't figure out any context from the input then give all the outcomes as an output with explaination.

Rules:
1. Follow the strict JSON output as per output schema
2. Always perform one step at a time and wait for next input

Output format: 
{ step: "string", content: "string" }

Example: 
Input: What is greater 9.8 or 9.11?
Output: { step: "analyze", content: "User is asking for which number is greater 9.8 or 9.11" }
Output: { step: "think", content: "By the mathematics, it's clear that 9.8 is greater than 9.11 because 9.8 means 9.80. But wait... in the context of book 9.11 is greater than 9.8 because we don't write it as 9.80 in this context. Let me think if there're any more context or the possible solutions"}
Output: { step: "think", content: "No more possible outcomes found" }
Output: { step: "choose", content: "There's no context given by the user. We can't figure out if user is asking for mathematical context or book context. So, we will give both the outcomes." }
Output: { step: "result", content: "9.8 is greater than 9.11 in mathematical context but 9.11 is greater than 9.8 in book context" }
"""

# Empty list to store conversation contents
contents = []

# Initial user input to start the conversation
query = input("Ask Anything > ") # e.g. "WWhat is the meaning of "fine"?", "What is greater 9.8 or 9.11?"
contents.append(
    types.Content(
        role="user",
        parts=[
            types.Part.from_text(text=query)
        ]
    )
)

# Chain of Thoughts + Self Consistency Prompting
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
