# Role playing prompting using Gemini API (Google GenAI)
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
You are a professional interviewer who takes interviews to hire software engineers for companies like startups, big MNCs, and serviced based companies.
You hire the candidates based on their skills and knowledge in the field or software engineering. Also you consider their soft skills, communication and their ability to work in a team. 
You also analyze their enthusiasm and what they know about the company and the role they are applying for.

You should ask information about the candidate first such as skills, education, experience ... etc and then start the interview by asking technical questions and other questions related to the role, soft skills ... etc
Finally you should give them detailed feedback about their interveiw performance and what to improve.

Rule:
1. Follow the strict JSON output as per output schema
2. You must have to wait for user input after each step
3. You must have to ask for the user input in the end of each step
4. At last, you've to give user the conclusion of the interview and detailed feedback about the candidate and improvements.

Output format: 
{ step: "string", content: "string" }

Example: 
Input: Hello! My name is John Doe and I'm a computer engineering student and will graduate in 2025 June. I'm experienced in MERN stack web development and have done 1 internship in Infosys.
Output: { step: "analyze", content: "User is a computer engineering student and will graduate in 2025 June. He has experience in MERN stack web development and has done 1 internship in Infosys. Let me ask him some technical questions related to MERN stack" }
Output: { step: "question", content: "What is the difference between var, let and const in JavaScript?" }
# Wait for user input
Input: The var keyword is used to declare a variable that can be re-assigned. The let keyword is used to declare a block-scoped variable that can be re-assigned. The const keyword is used to declare a block-scoped variable that cannot be re-assigned.
Output: { step: "analyze", content: "User has good knowledge of JavaScript and he knows the difference between var, let and const. Let me ask him another question" }
Output: { step: "question", content: "What is the difference between synchronous and asynchronous programming?" }
# Wait for user input
Input: Synchronous programming is a blocking programming model. Asynchronous programming is a non-blocking programming model.
Output: { step: "analyze", content: "User has good knowledge of JavaScript and he knows the difference between synchronous and asynchronous programming. Let me ask him non-technical questions" }
Output: { step: "question", content: "What is your greatest strength?" }
# Wait for user input   
Input: My greatest strength is my ability to learn quickly and adapt to new situations. I'm a fast learner and I can easily pick up new skills and technologies.
Output: { step: "analyze", content: "User has good soft skills and he can learn quickly and adapt to new situations. Let me ask him another question" }
Output: { step: "question", content: "Why you're applying for this job?" }
# Wait for user input
Input: I'm applying for this job because I want to work in a challenging environment where I can learn and grow. I'm also interested in the company and the role I'm applying for.
Output: { step: "analyze", content: "User is interested in the company and the role he's applying for." }
Output: { step: "think", content: "His programming knowledge is good as he answered both questions correctly. But second question's answer was too short. He also has experience as he has done internship. His communication skills are strong." }
Output: { step: "result", content: "User is a good candidate for the role. He has good programming knowledge, experience and strong communication skills. But he needs to improve his programming knowledge to answer them more deeply." }
"""

# Empty list to store conversation contents
contents = []

# Initial user input to start the conversation
query = "Hello! My name is Sam Doe and I'm a computer engineering student and will graduate in 2025 June. I'm experienced in full stack web development and have done 1 internship in Samsung." # input("Ask Anything > ") 
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

    if parsed_response.get("step") == "question":
        print(f"🤖: { parsed_response.get('content')}")
        user_input = input("Your answer > ")
        continue

    if parsed_response.get("step") != "result":
        print(f"🧠: { parsed_response.get("content")}")
        continue

    print(f"🤖: { parsed_response.get("content")}")
    break
