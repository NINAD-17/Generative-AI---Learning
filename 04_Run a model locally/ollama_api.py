from fastapi import FastAPI
from ollama import Client

app = FastAPI()
client = Client(
    host="http://localhost:11434",  # it tells Client - where the ollama API is running
)

client.pull("gemma3:1b")  # pull the model from ollama API

@app.post("/chat")
def chat(message): # we've to forward all the messages to the ollama API
    response = client.chat(model="gemma3:1b", messages=[
        { "role": "user", "content": "Hey there!" }
    ])

    return response["message"]["content"]  # return the response from ollama API