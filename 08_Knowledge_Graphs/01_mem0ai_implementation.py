import os
from dotenv import load_dotenv
from mem0 import Memory
from google import genai
from google.genai import types

# environmental variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
QDRANT_HOST=os.getenv("QDRANT_HOST")
NEO4J_URL=os.getenv("NEO4J_URI")
NEO4J_USERNAME=os.getenv("NEO4J_USERNAME")    
NEO4J_PASS=os.getenv("NEO4J_PASS")

# configuration for embedder
config = {
    # which embedder to use
    "embedder": {
        "provider": "gemini",
        "config": {
            "model": "models/text-embedding-004",
            "api_key": GEMINI_API_KEY,
            # "embedding_dims": 1536
        }
    },
    # which LLM to use
    "llm": {
        "provider": "gemini",
        "config": {
            "model": "gemini-2.0-flash-001",
            "api_key": GEMINI_API_KEY
        }
    },
    # which vector store to use
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": QDRANT_HOST,
            "port": 6333,
            "embedding_model_dims": 768
        }
    },
    # which graph store to use
    "graph_store": {
        "provider": "neo4j",
        "config": {
            "url": NEO4J_URL,
            "username": NEO4J_USERNAME,
            "password": NEO4J_PASS
        }
    }
}

# create memory instance
mem_client = Memory.from_config(config)

# initialize gemini
client = genai.Client(
    api_key = GEMINI_API_KEY,
)

def convert_message_to_openai_format(message: types.Content) -> dict: 
    return { "role": message.role, "content": message.parts[0].text }

def chat(message):
    # retrieve existing memory
    mem_result = mem_client.search(query = message, user_id = "person_123")

    memories = "\n".join([m["memory"] for m in mem_result.get("results")])
    print("memories: ", memories)

    SYSTEM_PROMPT = f"""
        You are a Memory-Aware Fact Extraction Agent, an advanced AI designed to
        systematically analyze input content, extract structured knowledge, and maintain an
        optimized memory store. Your primary function is information distillation
        and knowledge preservation with contextual awareness.

        Tone: Professional analytical, precision-focused, with clear uncertainty signaling
        
        Memory and Score:
        {memories}
    """

    messages = [
        types.Content(
            role = "user",
            parts = [
                types.Part.from_text(text = message)
            ]
        )
    ]

    result = client.models.generate_content(
        model = "gemini-2.0-flash",
        config = types.GenerateContentConfig(
            system_instruction = SYSTEM_PROMPT
        ),
        contents = messages
    )

    messages.append(
        types.Content(
            role = "model",
            parts = [
                types.Part.from_text(text = result.text)
            ]
        )
    )

    # convert all messages to mem0 format - { "role": "", "content": "" }
    mem_ready_messages = [
        convert_message_to_openai_format(message) 
        for message in messages
    ]

    print("Mem Ready Messages: ", mem_ready_messages)

    mem_client.add(mem_ready_messages, user_id = "person_123")

    return result.text

while True:
    message = input("Message > ")
    print(f"BOT: {chat(message)}")