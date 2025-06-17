# Simple weather agent
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
import json
import requests

# Load Environment Variables
load_dotenv()

# Get API Key from .env file
api_key = os.getenv("GEMINI_API_KEY")

# Initialize Google GenAI Client
client = genai.Client(
    api_key=api_key,
)

# Functions to be used in the agent
def get_weather(city: str) -> str:
    print("🔨 Tool Called: get_weather", city)

    url = f"https://wttr.in/{city}?format=%C+%t"
    response = requests.get(url)

    if response.status_code == 200:
        return f"The weather in {city} is {response.text}"
    
    return "Something went wrong while fetching the weather data."

def run_command(command: str):
    print("🧑‍💻 Running Command:", command)
    result = os.system(command=command)
    print(f"Command '{command}' executed. Result: {result}")
    return result

# Available Tools
available_tools = {
    "get_weather": {
        "func": get_weather,
        "description": "Takes a city name as an input and returns current weather for the city."
    },
    "run_command": {
        "func": run_command,
        "description": "Takes a command as an input and runs it on the system."
    }
}

# Available Tools data for system prompt (function: description)
available_tools_data = ""

for key, value in available_tools.items():
    print(key, value)
    available_tools_data += f"{key}: {value["description"]}\n"

# Define the system prompt for the model
system_prompt = f"""
You're an helpful AI assistance who's specialized in resolving user query.
You work on start, plan, action, observe mode.
For the given user query and available tool, plan the step by step execution.
Based on the planning select the relevant tool from the available tools and perform an action to call the tool.
Wait for the observation and based on the observation from the tool call resolve the user query.

Rules: 
1. Follow the output JSON Format.
2. Always perform one step at a time and wait for next input.
3. Carefully analyze the user query.
4. Before calling the tool, make sure that you don't have the answer already in previous messages. If you've knowledge from previous messages then use that knowledge instead of calling the tool.

Output format:
{{ step: "string", "content": "string", "function": "name of function if step is action", "input": "the input parameter for the function" }}

Available Tools:
{available_tools_data}

Example: 
User Query: What is the weather of Seoul?
Output: {{ "step": "plan", "content": "The user is interested in weather data of Seoul which is a city and capital of South Korea." }}
Output: {{ "step": "plan", "content": "From the available tools I should call get_weather" }}
Output: {{ "step": "action", "function": "get_weather", "input": "Seoul" }}
Output: {{ "step": "observe", "output": "25 degree celsius" }}
Output: {{ "step": "resolve", "content": "The weather of Seoul is 25 degree celsius." }}
"""

print(f"*System Prompt: {system_prompt}")

# Empty list to store conversation contents
contents=[]

while True: # Taking user query in a loop
    query = input("WEATHER INFORMATION ->\nEnter Query > ") # Create a file "magic.txt" in current directory # What is the weather of Seoul?
    contents.append(
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=query)
            ]
        )
    )

    while True:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
            ),
            contents=contents
        )

        parsed_response = json.loads(response.text)
        contents.append(
            types.Content(
                role="model",
                parts=[
                    types.Part.from_text(text=json.dumps(parsed_response))
                ]
            )
        )

        if parsed_response.get("step") == "plan":
            print(f"🧠: {parsed_response}")
            continue

        if parsed_response.get("step") == "action":
            print(f"🧠: {parsed_response}")
            tool_name = parsed_response.get("function")
            tool_input = parsed_response.get("input")

            if available_tools.get(tool_name, False) != False: 
                output = available_tools[tool_name].get("func")(tool_input)
                contents.append(
                    types.Content(
                        role="model",
                        parts=[
                            types.Part.from_text(text=json.dumps({
                                "step": "observe",
                                "output": output
                            }))
                        ]
                    )
                )
                continue

        if parsed_response.get("step") == "resolve":
            print(f"🤖: {parsed_response.get('content')}")
            break

    if query.lower() == "exit": 
        print("Exiting the weather agent.")
        break