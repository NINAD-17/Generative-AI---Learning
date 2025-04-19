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

# Generate response using the model
response = client.models.generate_content(
    model="gemini-2.0-flash",
    config=types.GenerateContentConfig(
        system_instruction=system_prompt,
        response_mime_type="application/json",
        # max_output_tokens=100,
        # temperature=0.5 # Balanced creativity and accuracy
    ),
    contents=[
        # Initial user input to start the conversation
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="""What is 3 + 4 * 5?""")
            ]
        ),
        # The model will follow the steps as per the system prompt
        # 1. Model analyzed the input
        types.Content(
            role="model",
            parts=[
                # json.dumps to stringify the JSON object
                types.Part.from_text(text=json.dumps({
                    "step": "analyze",
                    "content": "The user is asking to solve a mathematical expression that involves addition and multiplication. I need to remember the order of operations (PEMDAS/BODMAS)."
                }))
            ]
        ),
        # Sent the input by attaching response of model (initial response + 1st step response)
        # 2. Model thought about the input
        types.Content(
            role="model",
            parts=[
                types.Part.from_text(text=json.dumps({
                    "step": "think",
                    "content": "According to the order of operations, multiplication should be performed before addition. Therefore, I need to first multiply 4 and 5, and then add the result to 3."
                }))
            ]
        ),
        # Sent the input by attaching response of model (initial response + 1st step response + 2nd step response)
        # 3. Model output the result of multiplication
        types.Content(
            role="model",
            parts=[
                types.Part.from_text(text=json.dumps({
                    "step": "output",
                    "content": "23"
                }))
            ]
        ),
        # Sent the input by attaching response of model (initial response + 1st step response + 2nd step response + 3rd step response)
        # 4. Model validated the output
        types.Content(
            role="model",
            parts=[
                types.Part.from_text(text=json.dumps({
                    "step": "validate",
                    "content": "4 * 5 = 20, and 3 + 20 = 23. The calculation seems correct."
                }))
            ]
        ),
        # Final step generated the output and explaination
        # Note: If you attach, final step as well ({ step: "result" }), then it might start all the steps again, means give response of step "analyze".
        # You can try this by uncommenting the below code.
        # types.Content(
        #     role="model",
        #     parts=[
        #         types.Part.from_text(text=json.dumps({
        #             "step": "result",
        #             "content": "3 + 4 * 5 = 23. This is calculated by first multiplying 4 and 5, which equals 20. Then, adding 3 to 20, which equals 23."
        #         }))
        #     ]
        # )
    ]
) 

# Print the response
print(response.text)