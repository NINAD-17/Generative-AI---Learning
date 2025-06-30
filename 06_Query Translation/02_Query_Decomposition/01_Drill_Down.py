# Progressive Query Decomposition (Drill Down Prompting) - Less Abstract

# To Do:
# 1. Take user query
# 2. Break query into logical sub-questions
# 3. Sequential fetch chunks for each query in such a way that
#           query + previous query's chunk -> query + previous query's chunk -> ...
#           it means, append the previous query's context (chunk) to the next query run
# 4. Synthesize final answer based on all retrieved chunks

# Import Packages
import os
from dotenv import load_dotenv
from pathlib import Path
import json
from google import genai
from google.genai import types
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import ingest
import retrieve

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
pdf_path = Path(__file__).parent / "../data/Atomic Habits.pdf"
pdf_name = pdf_path.name

should_ingest_pdf, collection_name = ingest.should_ingest(pdf_path)
if should_ingest_pdf:
    ingest.ingest_pdf_to_qdrant(pdf_path, collection_name, embedding)
else:
    # use the existing collection_name as-is, and skip re-ingesting
    pass

# Retrieval
n_queries = 3 # number of queries to generate (default)
max_chunks = 10 # number of chunks to retrieve for each query from DB
generated_queries = [] # to store all generated queries

system_instructions = f"""
<goal>
You are an intelligent AI assistant specialized in extracting and synthesizing information from a provided PDF titled '{pdf_name}'.
Your task is to use the Drill Down method (also called Progressive Query Decomposition) to break the user's original question into a series of logically ordered sub-questions.
This will help make the user’s query less abstract by gradually unfolding the foundational knowledge leading to their request.

The flow is:
- Start with basic conceptual questions.
- Build upward in complexity and specificity.
- End with the user's original query.

Your output should help the system retrieve diverse and complete information from the document, leading to a final, well-informed answer.
</goal>

<steps>
1. think - Evaluate the completeness, clarity, and specificity of the user’s query.
2. ask - If the query is too vague, too broad, or ambiguous, ask a clarifying follow-up question.
3. generate_queries - For well-formed queries, generate {n_queries} meaningful sub-questions that escalate conceptually toward the user’s intent.
4. final_answer - After getting summarized text of all chunks, synthesize a helpful and accurate response using the summaries.
</steps>

<rules>
1. If the user’s query is ambiguous or unclear, start with step = "think" or "ask".
2. If the query is answerable, proceed to step = "generated_queries" and create {n_queries} sub-questions + include the user's original query as the final one.
3. Ensure the sub-questions are ordered logically from fundamental → specific.
4. Avoid generating redundant or overlapping sub-questions. Each should expand the conceptual understanding of the topic.
5. Do not fabricate content; always base reasoning and answers on chunks retrieved from the PDF.
6. If no relevant chunks are retrieved for a given sub-query, report that clearly in the final answer step.
7. If the PDF doesn't contain relevant information at all, respond with a polite message like:
   "This PDF doesn't seem to contain relevant content for your query. Would you like me to generate an answer using my pre-trained knowledge instead?"
8. Follow the JSON output format exactly as defined in <output>.
9. Only follow literal instructions for <output>; you are free to paraphrase anywhere else.
</rules>

<note>
Do not treat the user's query as a list of keywords. Think like a tutor guiding someone step-by-step toward deeper understanding.
</note>

<output>
{{ "step": "think", "content": "describe your thinking about the query" }}
{{ "step": "ask", "content": "clarifying question if needed" }}
{{ "step": "generated_queries", "queries": ["query1", "query2", "query3", ...], "original_query": "user's original query" }}
{{ "step": "final_answer", "answer": "Your final synthesized response" }}
</output>

<answer_format>
- Format using Markdown markup.
- Use # for the main topic.
- Use ## for each main section or supporting point.
- Use **bold** for important terms or definitions.
- Use regular text for explanations.
- For lists, use bullet or numbered points.
- After answering, suggest 3 follow-up questions (as bullets).
- Adjust length based on user preference (short/medium/long).
- By default, use medium length and include a brief overview, explanation, examples, and conclusion.
</answer_format>

<examples>
Example: 
User Query -> What is Machine Learning?
Assistant -> {{ "step": "think", "content": "Query is complete and not ambiguous, so I should generate queries." }}
Assistant -> {{ "step": "generated_queries", "queries": ["What is Machine?", "What is "Learning?"], "original_query": "What is Machine Learning?" }}
Now, system will retrieve chunks and give answer for all queries to you
User -> {{ "all_query_answers": "you'll get summary with all the queries", "user_original_query": "What is Machine Learning?" }}
Assistant -> {{ "step": "final_answer", "answer": "Your final answer by reviewing all_query_answers and user_original_query" }}
</examples>
"""

summary_instructions = f"""
You are a context summarizer.
You will get data as {{ "step": "summarize", "instruction": "instruction for you", "query": "user's query", "chunks": [ all the retrieved chunks from vector DB ]}}
You've to follow the instructions. 
Read the query and the data get from the chunks. 
Summarize the answer.

Output Format: {{ "step": "summary_response", "summary": "summary answer of user query and chunks" }}
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

def send_to_llm(system_prompt, input_context):
    response = client.models.generate_content(
        model = "gemini-2.0-flash",
        config = types.GenerateContentConfig(
            system_instruction = system_prompt,
            response_mime_type = "application/json",
        ),
        contents = input_context
    )

    return response

def main():
    while True:
        user_input("Ask anything on your PDF -> ")

        while True:
            response = send_to_llm(system_instructions, contents)

            # Parse and Append the response get from LLM
            try:
                parsed_response = json.loads(response.text) # parse the response from JSON string to python dict
            except json.JSONDecodeError:
                print("⚠️ LLM did not return valid JSON:", response.text)
                continue

            # Attach response to the context - 'content'
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
                queries = parsed_response.get("queries", [])
                user_original_query = parsed_response.get("original_query")
                
                print("GENERATED QUERIES: ") 
                for i, query in enumerate(queries, start=1):
                    print(f"\t{i}. {query}")
                print()

                generated_queries.extend(queries) # extend - takes every item from queries and appends it to generated_queries

                retrieved_chunks = [] # [[{}, {}, ...], [], [], ...]
                summary_of_chunks = {} # Format: { generated_query: summary of chunks }
                
                # Retrieve chunks for each query and generate summary of it
                for index, generated_query in enumerate(generated_queries):
                    # Format generated_query to add previous query's retrieved chunk summary
                    enriched_query = ""
                    if index == 0:
                        enriched_query = generated_query
                    else:
                        prev_summary = ""
                        for prev_q, prev_sum in summary_of_chunks.items():
                            prev_summary += prev_sum + "\n"
                        enriched_query = f"{generated_query}\nPrevious summaries:\n{prev_summary}"

                    print("Enriched query: ", enriched_query)

                    # Retrieve chunks for enriched_query = current query + prev query's summary
                    retrieved_query_chunks = retrieve.retrieve_relevant_chunks(enriched_query, max_chunks, embedding, collection_name, score_threshold = 0.7)
                    retrieved_chunks.extend(retrieved_query_chunks)

                    input_for_summarizer = json.dumps({
                        "step": "summarize",
                        "instruction": "For the given query there're fetched relevant chunks. Use those chunks and make a summarized answer for the query by using the chunks. ",
                        "query": generated_query,
                        "chunks": retrieved_query_chunks
                        # "prev_summary": summary_of_chunks # if you want to give previous summary {..., query: summary, ...} then you can. But here I thought we don't need to send it as we're retrieving documents based on summary so it'll have a context of all previous summary 
                    })

                    summary_context = []
                    summary_context.append(
                        types.Content(
                                role="user",
                                parts=[types.Part.from_text(text = input_for_summarizer)]
                        )
                    )
                    
                    # Generated summary
                    summary_response = send_to_llm(summary_instructions, summary_context)
                    parsed_summary_response = json.loads(summary_response.text)

                    # Store the summary into summary_of_chunks dictionary as { query: summary }
                    try:
                        summary_of_chunks[generated_query] = parsed_summary_response["summary"].strip()
                    except Exception:
                        summary_of_chunks[generated_query] = ""

                # Send user's original query and all the generated query answers to the LLM with our main context i.e. 'contents'
                contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text = json.dumps({
                            "all_query_answers": summary_of_chunks,
                            "user_original_query": user_original_query
                        }))]
                    )
                )
                
                continue

            if step == "final_answer":
                print("🤖 FINAL ANSWER: \n", parsed_response.get("answer"), "\n")
                break

if __name__ == "__main__":
    main()