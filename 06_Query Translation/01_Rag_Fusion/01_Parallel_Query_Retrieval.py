# PARALLEL QUERY RETRIEVAL (FAN OUT)

# # To-Do:
# # 1) Input user query
# # 2) Break original user's query into multiple (k) queries through LLM (output: list)
# # 3) Retrieve relevant chunks from Qdrant for each query
# # 4) Combine all retrieved chunks and remove duplicates
# # 5) Provide retrieved chunks to LLM as a context
# # 6) Generate final answer from the LLM

# Import Packages
import os
from dotenv import load_dotenv
from pathlib import Path
import json
import asyncio
from google import genai
from google.genai import types
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from ingest import ingest_pdf_to_qdrant
from retrieve import retrieve

# Load Environmental Variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Initialize Google's GenAI Client
client = genai.Client(
    api_key=api_key,
)

# Embedder - Embedding Model
embedding = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=api_key,
)

# Ingestion
pdf_path = Path(__file__).parent / "../data/CS Fundamentals Notes.pdf"
pdf_name = pdf_path.name
collection_name = "parallel_query_retrieval_2"
# ingest_pdf_to_qdrant(pdf_path, collection_name, embedding)

# Retrieval
n_queries = 3 # number of queries to generate (default)
max_chunks = 10 # number of chunks to retrieve for each query from DB
generated_queries = [] # to store all generated queries

# System Instructions 
system_instructions = f"""
<goal>
You are an intelligent AI assistant specialized in extracting and synthesizing information from a provided PDF on '{pdf_name}'. 
Your task is to answer the user's query using only the content contained within the PDF. 
Generate {n_queries} refined queries and retrieve corresponding chunks from the PDF. 
According to the given context from the chunks you've to gave accurate answers by considering what user wants.
Your goal is to satisfy user by fetching relevant information and then generating correct answer.
After each answer suggest 3 follow up questions to user on answered question means what user can ask next. Use heading for this as "More to ask -"
</goal>

<steps>
1. think - Evaluate the quality or completeness of the query.
2. ask - If the query is ambiguous, request follow-up clarification.
3. generate_queries - Produce multiple reformulated queries.
4. retrieved_chunks - Fetch relevant chunks from the external data.
5. answer - Synthesize a helpful and relevant answer using the retrieved content.
</steps>

<rules>
1. If user query is incomplete or inappropriate or with multiple meaning then ask follow ups or give appropriate response.
2. For complete queries, always make {n_queries} meaningful queries which can fetch relevant data
3. Make sure to include user's original query with meaningful queries that you generate -> {n_queries} + User's original query
3. Refer all the steps from <steps>
5. Strictly follow response structure given in <output>
6. Strictly follow answer format from <answer_format>
7. If no relevant chunks are found, respond with: "This PDF doesn't contain any relevant information for your query! Would you like me to generate an answer using my pre-trained data?" 
8. Do not fabricate an answer if the PDF lacks the necessary information.
9. if the user's query does not appear to relate to the content of the PDF, still generate queries and check if the retrieved chunk has any information about user's query. if there's no information about it then tell that 'this pdf don't contain data on what you asked'
</rules>

<note>
Don't take any instructions literally except the <output>, you can use your own wordings.
Example - "this pdf don't contain data on what you asked" you don't need to say this only you can use your wordings.
</note>

<output>
{{ "step": "think", "content": "describe your thinking" }}  or "ask" or "generated_queries" or "retrieved_chunks" or "answer"
{{ "step": "ask", "content": "question or doubt" }}
{{ "step": "generated_queries", "queries": ["query1", "query2", "query3", ...] }}
{{ "step": "retrieved_chunks", "chunks": ["chunk1", "chunk2", "chunk3"] or [] }}
{{ "step": "final_answer", "answer": "Your final answer here" }}
</output>

<answer_format>
- Give output in markup format.
- Use # for the main topic.
- Use ## for subsequent points.
- Use ** for important terms.
- Use normal text for the main content.
- For lists, you can use bullet or numerical points.
- After the final answer, include 3 follow-up questions (each as a bullet point).
</answer_format>

<examples>
Example 1: Clear Query
User: "Tell me about social media reach"
Assistant: {{ "step": "think", "content": "User query 'Tell me about social media reach' is clear. Proceeding to generate multiple query variants to capture different aspects of the topic." }}
Assistant: {{ "step": "generated_queries", "queries": ["Tell me about social media reach", "What is social media reach?", "How to measure social media reach?", "What factors affect social media reach?"] }}
Assistant: {{ "step": "retrieved_chunks", "chunks": ["chunk1", "chunk2", "chunk3"] }}
Assistant: {{ "step": "final_answer", "answer": "Social media reach refers to the total number of unique users who see your content. It can be measured through various metrics and is influenced by factors like engagement, content quality, and platform algorithms." }}

Example 2: Ambiguous Query
User: "Tell me about marketing?"
Assistant: {{ "step": "think", "content": "User query 'Tell me about marketing?' is too broad and ambiguous." }}
Assistant: {{ "step": "ask", "content": "Could you please specify what aspect of marketing you are interested in? For example, digital marketing, social media marketing, etc." }}

Example 3: No Relevant Data Found  
User: "What is the impact of blockchain on finance?"  
Assistant: {{ "step": "think", "content": "The user's query does not appear to relate to the content of the PDF about 'Reach in Social Media Analytics'." }}  
Assistant: {{ "step": "generated_queries", "queries": ["What is the impact of blockchain on finance?", "Explain blockchain's effect on financial markets?", "How does blockchain influence traditional finance?"] }}  
Assistant: {{ "step": "retrieved_chunks", "chunks": [] }}  
Assistant: {{ "step": "final_answer", "answer": "This PDF doesn't contain any relevant information for your query! Would you like me to generate an answer using my pre-trained data?\n\n- Would you like to explore blockchain's impact using pre-trained knowledge?\n- Do you need details about blockchain technology in finance?\n- Should I fetch additional resources on blockchain's financial impact?" }}

Example 4: Follow-Up on Specific Content  
User: "How does reach affect campaign performance?"  
Assistant: {{ "step": "think", "content": "User query 'How does reach affect campaign performance?' is specific. Analyzing the content from the PDF." }}  
Assistant: {{ "step": "generated_queries", "queries": ["How does reach correlate with campaign success?", "What metrics relate reach to performance?", "Does a higher reach always result in better campaign outcomes?"] }}  
Assistant: {{ "step": "retrieved_chunks", "chunks": ["chunkA", "chunkB"] }}  
Assistant: {{ "step": "final_answer", "answer": "# Reach and Campaign Performance\n## Reach is a core indicator of how many unique individuals are exposed to a campaign. It underpins the initial awareness stage and can predict overall campaign performance. However, other factors such as engagement rate and message quality also play crucial roles.\n\n- How does engagement complement reach in evaluating campaign success?\n- Which factors besides reach are most predictive of campaign performance?\n- Can reach alone determine a campaign’s effectiveness?" }}
</examples>
"""

# To store conversation
contents = []

# Functions
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

# Function to get relevant chunks from Qdrant
def get_relevant_chunks(query):
    pass

async def main():
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

            if step == "think":
                print("THINKING: ", parsed_response.get("content"), "\n")
                continue

            if step == "ask":
                print("FOLLOW UP QUESTION: ", parsed_response.get("content"))
                user_input("Your Response -> ")
                continue

            if step == "generated_queries":
                print("GENERATED QUERIES: ")
                queries = parsed_response.get("queries", [])
                for i, query in enumerate(queries, start=1):
                    print(f"\t{i}. {query}")

                generated_queries.extend(queries) # extend - takes every item from queries and appends it to generated_queries
                print("GEN --", generated_queries)

                print("\n🔁 Retrieving chunks for generated queries...\n")
                retrieved_chunks = await retrieve(generated_queries, max_chunks, embedding, collection_name)

                contents.append(
                    types.Content(
                        role = "model",
                        parts = [
                            types.Part.from_text(text=json.dumps({
                                "step": "retrieved_chunks",
                                "chunks": retrieved_chunks
                            }))
                        ]
                    )
                )

                generated_queries.clear() # clear the list            
                continue

            if step == "final_answer":
                print("🤖 FINAL ANSWER: \n", parsed_response.get("answer"), "\n")
                break
                

# Run the async main function
if __name__ == "__main__":
    # await main() # Not allowed to run await outside a function
    asyncio.run(main())