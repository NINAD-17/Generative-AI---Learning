# Logical Routing

# Note: Given is the simple implementation of LLM routing. 
#       Currently it's not handling any cases. Description of model is also not good so it can give wrong answer.
#       It's only for learning how routing works.

# Import Packages
import os
from dotenv import load_dotenv
from pathlib import Path
import json
from google import genai
from google.genai import types

# Load Environmental Variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Initialize Google's GenAI Client
client = genai.Client(
    api_key=api_key,
)

# Available Models
all_models = {
    "gemini-2.5-flash-lite":
        "Gemini 2.5 Flash-Lite is Google's fastest and most cost-effective model, ideal for general-purpose queries, day-to-day questions, quick information lookup, translation, summarization, basic planning, and simple creative writing. It supports multimodal input (text, images, audio, video) and is perfect for high-volume, low-complexity tasks where speed and efficiency matter most.",

    "deepseek-coder-v2-lite-instruct":
        "DeepSeek Coder V2 Lite Instruct is a lightweight, open-source code model optimized for simple coding tasks, code completion, basic debugging, and straightforward scripting in 338 programming languages. Use this model for beginner-level programming help, code snippets, and fast, cost-efficient code generation.",

    "deepseek-coder-v2":
        "DeepSeek Coder V2 is a state-of-the-art, code-focused model designed for complex coding challenges, advanced code generation, multi-file projects, code insertion, code review, and mathematical reasoning. It excels in technical programming interviews, algorithmic problems, and large codebase analysis, supporting long context windows and 338 languages.",

    "gpt-4o":
        "GPT-4o is OpenAI's flagship multimodal model, best for advanced research, technical analysis, deep document understanding, planning, organization, creative writing, technical Q&A, and all multimodal tasks (text, image, audio, video). It delivers real-time performance, excels at summarization, brainstorming, and is highly capable for complex reasoning, planning, and productivity workflows.",

    "gemini-2.5-pro":
        "Gemini 2.5 Pro is Google's most advanced reasoning and multimodal model, ideal for tasks requiring deep understanding, such as scientific research, STEM problem-solving, large-scale data analysis, processing entire code repositories, and handling very long documents or complex multimodal queries. Use this model for the most challenging research and reasoning tasks."
}

model_list = ""
for index, (model, desc) in enumerate(all_models.items(), start = 1):
    model_list += f"{index}. {model}:\n{desc}\n\n"

# print(model_list)

# System Instructions
SYSTEM_PROMPT = f"""
<goal>
- You're an intelligent AI assistant proficient in decision making.
- You'll have a list of available AI models in <model_list> with their use.
- Your task is to pick a model which is best fit for the user's query.
</goal>

<model_list>
{model_list}
</model_list>

<steps>
1. Read the user’s query.
2. Decide which model can solve it correctly without overkill.
3. Return JSON: {{ "model": "model_name", "reason": "short reason: why this model fits best" }}
</steps>

<rules>
1. Follow all the steps
2. Give output strictly in JSON.
</rules>
"""

while True:
    user_query = input("Ask Anything -> ")

    if user_query == "exit":
        exit()

    response = client.models.generate_content(
        model = "gemini-2.0-flash",
        config = types.GenerateContentConfig(
            system_instruction = SYSTEM_PROMPT,
            response_mime_type = "application/json",
        ),
        contents = types.Content(
            role="user",
            parts=[
                types.Part.from_text(text = user_query)
            ]
        )
    )

    parsed_response = json.loads(response.text)
    choosed_model = parsed_response.get("model", "NULL")
    reason = parsed_response.get("reason", "NULL")
    print(f"For this query we'll choose {choosed_model} model because - {reason}\n")