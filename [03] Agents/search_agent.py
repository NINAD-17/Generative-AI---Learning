from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
import json
import requests

# Load Environment Variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
google_cse_api_key = os.getenv("GOOGLE_SEARCH_KEY")
custom_search_id = os.getenv("SEARCH_ENGINE_ID")

# Initialize Google GenAI Client
client = genai.Client(api_key = gemini_api_key)

# Define Tools
def google_search(query, search_limit = 5):
    print("Searching On Google: \n")
    params = {
        "q": query,
        "num": search_limit,
        "key": google_cse_api_key,
        "cx": custom_search_id
    }

    response = requests.get("https://customsearch.googleapis.com/customsearch/v1", params = params)
    parsed_response = response.json()
    items = parsed_response.get("items", []) # if there're no items found return an empty array
    return [{ "title": i["title"], "url": i["link"], "snippet": i["snippet"]} for i in items] # return an array of all fetched result - [title, url, snippet] only

# List of available tools
available_tools = {
    "google_search": {
        "function": google_search,
        "description": "Takes query and limit as an input and returns results in an array format which size is less than or equal limit"
    }
}

# Available Tools data for system prompt (function: description)
available_tools_data = ""

for key, value in available_tools.items():
    available_tools_data += f"{key}: {value['description']}"

# System Prompt
system_instructions = f"""
You're an intelligent AI assistant specialized in resolving user's query. 
You use different available tools to resolve user queries and satisfy their needs.
For the given user query and available tools, plan the step by step execution.
If you want any confirmation from user, or ask follow ups then you can using 'ask' step.
You read the user query, analyze it, think, again think until you don't understand the query correctly, use relevant tool from available tools if necessary, observe the output and then generate good satisfactory output.

Rules: 
1. Follow the output JSON Format.
2. Always perform one step at a time and wait for next input.
3. Carefully analyze the user query.
4. Before calling the tool, make sure that you don't have the answer already in previous messages. If you've knowledge from previous messages then use that knowledge instead of calling the tool.
5. When you call any tool, wait for the observation and based on the observation from the tool call resolve the user query.

Output format:
{{ step: <step_name>, "content": "...", "function": <tool_name>, "input": <tool_input> }}

Available Tools:
{ available_tools_data }

Examples: 
Query: What is the capital of Korea?
Model: {{ "step": "analyze", "content": "User is asking about capital of country Korea. But there're two Korea's in the world" }}
Model: {{ "step": "think", "content": "I know the capitals of both countries. Do I need to call any available tools for answer?" }}
Model: {{ "step": "think", "content": "This is static fact that I already. No need to call any tool for searching the answer" }}
Model: {{ "step": "resolve", "content": "There're two Koreas in the world. If you're asking for South Korea (Republic of Korea) then its capital is Seoul and capital of North Korea (Democratic People's Republic of Korea) is Pyongyang" }}

Query: Where I can watch india vs england test match
Model: {{ "step": "analyze", "content": "User wants to know about latest event. I might not have knowledge about this." }}
Model: {{ "step": "action", "content": "From the available tools I'll choose google_search tool.", "function": "google_search", "input": "Streaming platform for latest india vs england test match" }}
<-- Calling the tool -->
Model: {{ "step": "observe", "output": [{{'title': 'Where to watch Ind vs Eng test 2025?', 'snippet': 'watch ind vs eng 2025 test match on JioHotstar'}}] }}
MOdel: {{ "step": "resolve", "content": "You can watch Ind VS Eng test match on JioHotstar streaming platform " }}

Query: What is the launch date of 
Model: {{ "step": "analyze", "content": "User wants a launch date but hasn’t specified the product or mission. I'll need to ask it" }}
Model: {{ "step": "ask", "content": "Sure—could you please specify which product, mission, or service you’re asking about?"}}
Query: I am talking about Squid Game season 3
Model: {{ "step": "analyze", "content": "Got it! For this I'll need to search it on internet" }}
Model: {{ "step": "action", "content": "From the available tools I'll choose google_search tool", "function": "google_search", "input": "launch date of Squid Game season 3 kdrama" }}
<-- Calling the tool -->
Model: {{ "step": "observe", "output": [{{ "title": "Final season of Squid Game 3 will be releasing soon!", "snippet": "Squid Game 3 is set to launch on 27th June on Netflix" }}] }}
Model: {{ "step": "resolve", "content": "Launch date of squid game is 27th June and it'll be stream on Netflix" }}
"""

# To store conversation
contents = []

def user_input(input_param = "Ask Anything -> "):
    query = input(input_param)

    if query.lower() == "exit": 
        print("Exiting the search agent.")
        exit()  

    contents.append(
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=query)
            ]
        )
    )

while True:
        user_input("Ask Anything -> ")

        while True:
            response = client.models.generate_content(
                model = "gemini-2.0-flash",
                config = types.GenerateContentConfig(
                    system_instruction = system_instructions,
                    response_mime_type = "application/json",
                ),
                contents = contents
            )

            parsed_response = json.loads(response.text)
            contents.append(
                types.Content(
                    role = "model",
                    parts = [
                        types.Part.from_text(text=json.dumps(parsed_response))
                    ]
                )
            )

            if parsed_response.get("step") != "resolve":
                print(f"🧠: {parsed_response}")
                continue

            if parsed_response.get("step") == "action":
                tool_name = parsed_response.get("function")
                tool_input = parsed_response.get("input")

                if available_tools.get(tool_name, False) != False:
                    output = available_tools[tool_name].get("function")(tool_input)
                    contents.append(
                        types.Content(
                            role = "model",
                            parts = [
                                types.Part.from_text(text=json.dumps({
                                    "step": "observe",
                                    "output": output
                                }))
                            ]
                        )
                    )
                continue

            if parsed_response.get("step") == "ask":
                user_input(parsed_response.get("content"))
                continue

            if parsed_response.get("step") == "resolve":
                print(f"🤖: {parsed_response["content"]}")
                break # exit inner loop
          