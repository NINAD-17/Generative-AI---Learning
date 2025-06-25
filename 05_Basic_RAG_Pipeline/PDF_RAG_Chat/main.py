import os
from dotenv import load_dotenv
from pathlib import Path
import json
from google import genai
from google.genai import types
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from ingest import ingest_pdf_to_qdrant
from retrieve import retrieve_relevant_chunks

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

client = genai.Client(
    api_key=api_key,
)

embedding = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=api_key,
)

pdf_path = Path(__file__).parent / "../data/Reach - SMA.pdf"
collection_name = "rag_pdf_1"

# Run INGESTION
# ingest_pdf_to_qdrant(pdf_path, collection_name, embedding)
print("This PDF is already ingested")

# Run RETRIEVAL
def get_relevant_chunks(user_query):
    return retrieve_relevant_chunks(user_query, embedding, collection_name)

system_instructions = """
<goal>
You're an intelligent AI assistant.
You've access to an external tool called get_relevant_chunks.
Your work is to create a good query from the user's query to retrieve relevant chunks from external data.
</goal>

<steps>
1. Understand the user's request or follow-up.
2. If the query isn't clear then ask for follow-up using 'ask' step.
2. Extract a precise query that will help locate relevant knowledge from the external data.
3. Use `get_relevant_chunks` with that query.
4. Synthesize a helpful and relevant answer using the retrieved content.
</steps>

<rules>
1. Always execute according to the steps mentioned in <steps></steps>
2. For user query, always analyze what he wants and make a good query which can retrieve relevant chunks.
3. Follow the output JSON Format and only one object per message.
</rules>

<format_rules>
- Give output in markup format
- For the main topic that you're discussing use # markup
- For all subsequent points use ## markup
- For important terms use ** markup
- For regular answer use normal text
- For points you can use bullet or numerical points
</format_rules>

<tool_available>
get_relevant_chunks
</tool_available>

<output>
Always give satisfactory answer
Use all the context you've and user's query and give informative answer
According to user's preference you can give short, medium or very long answer
You can use tone according to the context or user's query
</output>

<examples>
Example 1: New user query
User: "Can you explain supervised learning?"
Assistant: 
    { "step": "think", "content": "User has asked about Supervised learning. It's a complete query which ask about something. So lets rephrase it for retrieving relevant chunks and call the tool" }
    { "step": "action", "tool": "get_relevant_chunks", "input": "Explain supervised learning with examples and its key characteristics" }
Tool: { "step": "observe", "content": [{ .... data .... }] }
Assistant: { "step": "output", "content": "suitable answer for query" }

Example 2: Incomplete user query
User: "Can you explain?"
Assistant: 
    { "step": "think", "content": "User wants explaination of something but the query is incomplete. I'll need to ask for clear query" }
    { "step": "ask", "content": "Seems like you missed something.", "input": "Can you please give me clearer query?" }
User: Can you explain supervised learning?
    { "step": "think", "content": "Now user has given the clearer query. Lets rephrase this query to get relevant chunks." }
    { "step": "action", "tool": "get_relevant_chunks", "input": "Explain supervised learning with examples and its key characteristics" }
Tool: { "step": "observe", "chunks": [{ .... data .... }] }
Assistant: { "step": "output", "content": "suitable answer for query" }
</examples>
"""

# To store conversation
contents = []

# Take input from user
def user_input(input_param = "Ask Anything -> "):
    query = input(input_param)
    print()

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
    user_input("Ask anything on your PDF -> ")

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
        try:
            parsed_response = json.loads(response.text) # parse the response from JSON string to python dict
        except json.JSONDecodeError:
            print("⚠️ LLM did not return valid JSON:", response.text)
            continue

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
            print("--MODEL IS ASKING FOLLOW UP QUESTION FOR CLEARER QUERY--")
            print(parsed_response.get("content"))
            formatted_input = parsed_response.get("input", "") + " -> "
            user_input(formatted_input)
            continue

        if step == "action":
            print("--MODEL IS CALLING TOOL FOR DATA RETRIEVAL--")
            tool_name = parsed_response.get("tool")
            tool_input = parsed_response.get("input")

            if tool_name == "get_relevant_chunks":
                print("Rephrased Query: ", tool_input)
                print("🔁 Retrieving chunks from vector database\n")

                output = get_relevant_chunks(tool_input)

                print("Retrieved Chunks ------------")
                for chunk in output:
                    print(f" • Page {chunk['page_num']} — {chunk['content'][:50]}...")
                print("-----------------------------\n")

                contents.append(
                    types.Content(
                        role = "model",
                        parts = [
                            types.Part.from_text(text=json.dumps({
                                "step": "observe",
                                "chunks": output
                            }))
                        ]
                    )
                )
            continue         

        if step != "output":
            print(f"--MODEL IS THINKING--\n🧠: {parsed_response.get("step")} - {parsed_response.get("content")}\n")
            continue

        # Final Answer
        print(f"--🤖: FINAL ANSWER--\n{parsed_response["content"]}\n")
        break 
