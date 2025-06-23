from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
import json
import requests
import logging # to report status, errors, warnings, and other diagnostic information

# Load Environment Variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY") # Gemini API
google_cse_api_key = os.getenv("GOOGLE_SEARCH_KEY") # Google Custom Search API
custom_search_id = os.getenv("SEARCH_ENGINE_ID") # Google Search Engine Id

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

    try:
        response = requests.get("https://customsearch.googleapis.com/customsearch/v1", params = params, timeout=10)
        parsed_response = response.json()
        items = parsed_response.get("items", []) # if there're no items found return an empty array
        return [
            {
                "title":   itm.get("title", ""),
                "url":     itm.get("link", ""),
                "snippet": itm.get("snippet", "")
            }
            for itm in items
        ] # return an array of all fetched result - [title, url, snippet] only
    except requests.exceptions.Timeout:
        logging.warning("Google search timed out.")
        return [{"error": "Search request timed out after 10 s"}]

    except requests.exceptions.HTTPError as e:
        logging.error(f"HTTP {response.status_code}: {response.text[:200]}")
        return [{"error": f"HTTP error {response.status_code}"}]

    except Exception as e:
        logging.exception("Unexpected search error")
        return [{"error": f"Unexpected error: {e}"}]

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
<goal>
You're an intelligent AI assistant.
You resolve the query, by using own knowledge and external data which can be get by available tools.
You understand the query, user preference and plan step by step execution. It includes analyzing the query, thinking multiple times, deciding on tool use and giving final answer.
</goal>

<steps>
1. analyze: for user query understanding
2. ask: for follow up question to the user
3. decide_tool: select the appropriate tool or method if needed
4. action: execute the choosen tool
5. observe: analyze the result or response returned by the tool
6. resolve: combine the retrieved data (if any) with the model’s reasoning and language capabilities to generate the final answer for the user.
</steps>

<rules>
1. Only steps in the <steps></steps> are allowed
2. Follow the output JSON Format and only one object per message.
3. Always perform one step at a time and wait for next input.
4. After analyze, output decide_tool {{need_tool, reason}}
5. If need_tool == "no", go straight to resolve; If need_tool == "yes", output action, then wait for an observe message
Reuse information from earlier observe steps whenever possible.
6. If you already know the answer then don't use any tool by need_tool = "no"
</rules>

<available_tools>
{ available_tools_data }
</available_tools>

<output_format>
{{ step: <step_name>, "content": "...", "function": <tool_name>, "input": <tool_input> }}
</output_format>

<examples>
Example 1: static fact 
Query: What is the capital of Korea?
Model: {{ "step": "analyze", "content": "User is asking about capital of country Korea." }}
Model: {{ "step": "decide_tool", "need_tool": "no", "reason": "static fact" }}
Model: {{ "step": "resolve", "content": "There're two Koreas in the world. If you're asking for South Korea (Republic of Korea) then its capital is Seoul and capital of North Korea (Democratic People's Republic of Korea) is Pyongyang" }}

Example 2: time-sensitive (tool)
Query: Where can I watch India vs England Test match?
Model: {{ "step": "analyze", "content": "User needs current streaming info of match" }}
Model: {{ "step": "decide_tool", "need_tool": "yes", "reason": "time_sensitive" }}
Model: {{ "step": "action", "function": "google_search", "input": "India vs England Test match streaming platform" }}
Tool:  {{ "step": "observe", "output": [{{'title': 'Where to watch Ind vs Eng test 2025?', 'snippet': 'watch ind vs eng 2025 test match on JioHotstar'}}] }}
Model: {{ "step": "resolve", "content": "You can watch Ind VS Eng test match on JioHotstar streaming platform" }}

Example 3: missing details (ask)
Query: What is the launch date of 
Model: {{ "step": "analyze", "content": "User wants a launch date but hasn’t specified the product or mission. I'll need to ask it" }}
Model: {{ "step": "ask", "content": "Sure—could you please specify which product, mission, or service you’re asking about?"}}
Query: I am talking about Squid Game season 3
Model: {{ "step": "analyze", "content": "Got it!" }}
Model: {{ "step": "decide_tool", "need_tool": "yes", "reason": "time_sensitive" }}
Model: {{ "step": "action", "function": "google_search", "input": "What is the launch date of Squid Game season 3?" }}
Tool: {{ "step": "observe", "output": [{{ "title": "Final season of Squid Game 3 will be releasing soon!", "snippet": "Squid Game 3 is set to launch on 27th June on Netflix" }}] }}
Model: {{ "step": "resolve", "content": "Launch date of squid game is 27th June and it'll be stream on Netflix" }}

Example 4: reuse previous context/search
Context observe: "...Katseye created by HYBE & Geffen..."
Query: Who created Katseye?
Model: {{ "step":"analyze", "content":"Creator of Katseye"}}
Model: {{ "step":"decide_tool", "need_tool":"no", "reason":"Info exists in previous observe" }}
Model: {{ "step":"resolve","content":"Katseye was created by HYBE and Geffen Records." }}
</examples>
"""

# To store conversation
contents = []

# Take input from user
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
    user_input("Ask Anything -> ") # until the user exits the program, we'll show an input for them to talk to our agent.

    while True:
        response = client.models.generate_content(
            model = "gemini-2.0-flash",
            config = types.GenerateContentConfig(
                system_instruction = system_instructions,
                response_mime_type = "application/json",
            ),
            contents = contents
        )

        # Parse and Append the response get from LLM
        parsed_response = json.loads(response.text) # parse the response from JSON string to python dict
        contents.append(
            types.Content(
                role = "model",
                parts = [
                    types.Part.from_text(text=json.dumps(parsed_response))
                ]
            )
        )

        # Step wise actions
        step = parsed_response.get("step")

        if step == "ask":
            user_input(parsed_response.get("content"))
            continue

        if step == "action":
            # print("ALL MSG: \n", contents)
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

        if step not in "resolve":
            print(f"🧠: {parsed_response}")
            continue

        # Final Answer
        print(f"🤖: {parsed_response["content"]}")
        break # exit inner loop
          