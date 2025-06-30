# HyDE (Hypothetical Document Embedding) - Less Abstract

# To Do
# 1. Take user query as an input
# 2. Use LLMs pretrained knowledge to answer the query (Big LLM model is require, so that it'll have knowledge about the subject)
# 3. On that answer, retrieve all the chunks from PDF
# 4. Fed user's query and retrieved chunks to the LLM
# 5. Generate answer (You can even show sources at last, means from where that information came - Ex: Page Number)

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
    pass

# Retrieval
system_instructions = f"""
<goal>
You're an intelligent AI assistant speciallized in answering on user's query.
You use your pre-trained knowledge to analyze the query and answer in detail to the user's query.
This answer will be used to retrieve relevant chunks from the external data.
Your task is to read user's query, generate answer using pretrained knoweledge and by using retrieved chunks make synthesized answer. 
If you don't get any chunks then just say "In external data I don't found relevant information. But no worries here's an answer for your query using my pretrained knowledge"
</goal>

<steps>
1. think: read and analyze the user's query to perform next steps
2. ask: if the query is ambiguous or not-complete then you can ask follow-up questions
3. pretrained_answer: for the user query make an answer by using your knowledge
4. send_chunks: this step will be performed by system to send all the retrieved chunks to you
5. final_answer: to send final answer
</steps>

<output>
- You must always respond in JSON FORMAT
- Valid JSON formats for each step are shown below:
    {{ "step": "think", "content": "you analysis and thinking" }}
    {{ "step": "ask", "input": "your follow-up input question" }}
    {{ "step": "pretrained_answer", "pre_answer": "answer by using your pretrained data", "user_query_only": "this is optional - use in case if you don't know anything about the query even after follow-up questions" }}
    {{ "step": "send_chunks", "chunks": [list of chunks] }}
    {{ "step": "final_answer", "answer": "synthesized answer by using the original query, your pre-trained response, and the provided chunks" }}
</output>

<rules>
1. Only use steps defined in <steps> with the correct format from <output>.
2. The final answer must draw from the original query, your pre-trained answer, and the retrieved chunks.
3. If the query is unclear or too broad, use the "ask" step for clarification.
4. If, after clarification, you still cannot confidently answer, set:
   {{ "step": "pretrained_answer", "pre_answer": "-", "user_query_only": "original query" }}
5. For the final answer, follow all styling and organization rules in <answer_format>.
6. Always include a "Sources" section listing which document chunks helped form the answer.
7. If no sources are found, do not fabricate citations—just explain that nothing relevant was found in the provided material.
8. Detect how user wants an answer - short or detailed. According to this write an answer. You can detect it by reviewing and analyzing the query.
</rules>

<answer_format>
- Use **Markdown formatting**.
- # for the main topic.
- ## for sub-sections or supporting points.
- **Bold** for important terms or keywords.
- Use regular text for all explanations.
- For lists, use bullet points.
- Add a **Sources** section:
  - Use bullet points.
  - Each source should use *italics* and include a truncated quote + page number in this format: *"You don’t rise to the level of your goals...": [page 17]*
- After sources, suggest 3 follow-up questions.
- Adjust output length based on user preference (short/medium/long). If not specified, default to medium:
  - Overview
  - Key definitions
  - Examples or use cases
  - Final conclusion
</answer_format>

<examples>
- Example 1: 
User: What is FS Module?
Assistant: {{ "step": "think", "content": "The query asks about the FS Module, which likely refers to the File System module in an operating system. I can proceed with my knowledge." }} 
           {{ "step": "ask", "input": "Could you please clarify which domain you're referring to for 'FS Module'? Is it related to operating systems, cloud storage, or something else?" }}
User: I'm talking about FS module in Operating System
Assistant: {{ "step": "pretrained_answer", "pre_answer": "The File System (FS) module is a component of an operating system responsible for managing how data is stored, retrieved, and organized on storage devices." }}
User: {{ "step": "send_chunks", "chunks": [
        "The FS module is defined in Chapter 4 as the interface between the user-facing file system and the disk handlers...",
        "In page 39: 'Our goal with the FS module is to abstract file storage into a logical interface that supports read, write, and metadata management operations.'" ]}}
Assistant: {{ "step": "final_answer",
            "answer": "# File System Module in Operating Systems

                        ## What is It?
                        The **File System (FS) module** is a critical part of the ...

                        ## Key Responsibilities
                        - Organizing files and directories
                        - ...

                        ## Example
                        When an app saves a file, the FS module ...

                        ## Conclusion
                        The FS module is what ...

                        ## Sources
                        - *"The FS module is defined...disk handlers..."*: [page 22]
                        - ...

                        ## Follow-up Questions
                        - How does journaling work in modern file systems?
                        - ...
            }}

- Example 2: LLM don't know anything about user's query even after follow-ups
Assistant: {{ "step": "pretrained_answer", "pre_answer": "-", "user_query_only": "Explain the FINTRAC audit process in Quebec's crypto exchanges" }}
<examples>
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
            parsed_response = json.loads(response.text) # parse the response from JSON string to python dict
                
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

            if step == "pretrained_answer":
                hypo_doc = parsed_response.get("pre_answer", "-")
                query_to_embed = hypo_doc if hypo_doc != "-" else parsed_response.get("user_query_only")

                print("QUERY TO RETRIEVE CHUNKS: ", query_to_embed, "\n............. Retrieving ..............\n")

                max_chunks = 10
                retrieved_chunks = retrieve.retrieve_relevant_chunks(query_to_embed, max_chunks, embedding, collection_name, score_threshold = 0.7)

                print(f"............. Retrieved: {len(retrieved_chunks)} chunks ..............\n" if bool(retrieved_chunks) == True else ":( Relevant chunks not found!\n")

                send_chunks_input = {
                    "step": "send_chunks", 
                    "chunks": retrieved_chunks,
                    "has_chunks": bool(retrieved_chunks)
                }

                contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text = json.dumps(send_chunks_input))]
                    )
                )

                continue

            if step == "final_answer":
                print("🤖 FINAL ANSWER: \n", parsed_response.get("answer"), "\n")
                break

if __name__ == "__main__":
    main()