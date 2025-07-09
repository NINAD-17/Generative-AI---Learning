# Simple ChatBot: Route based on Coding Question (Good Model) and Simple Question (Mini Model)

# ---------------------- Packages ----------------------
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langsmith.wrappers import wrap_openai
from typing import Literal
from dotenv import load_dotenv
from openai import OpenAI
import os
from pydantic import BaseModel

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

client = wrap_openai(OpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
))

# ---------------------- Pydantic Schema ------------
class DetectCallResponse(BaseModel):
    is_question_ai: bool # AI should give response in this format only

class CodingAIResponse(BaseModel):
    answer: str

# ---------------------- State ----------------------
class State(TypedDict):
    user_message: str
    is_coding_question: bool
    ai_answer: str

# ---------------------- Define Nodes ----------------------
def detect_query(state: State):
    user_message = state.get("user_message")

    SYSTEM_PROMPT = """
    You are an AI assistant. Your job is to detect if the user's query is related
    to coding question or not.
    Return the response in specified JSON boolean only.
    """

    # Gemini Call with OpenAI SDK and getting structured response
    result = client.beta.chat.completions.parse(
        model = "gemini-2.5-flash-lite-preview-06-17",
        response_format = DetectCallResponse, # AI will return response in this format (key: value)
        messages = [
            { "role": "system", "content": SYSTEM_PROMPT },
            { "role": "user", "content": user_message }
        ]
    )

    # print(result.choices[0].message.parsed) # output: is_question_ai=False
    state["is_coding_question"] = result.choices[0].message.parsed.is_question_ai
    return state

# Conditional Edge
def route_edge(state: State) -> Literal["solve_coding_question", "solve_simple_question"]: # It needs to know about where it can go (all possible return values) that's why we used Literal
    is_coding_question = state.get("is_coding_question")

    if is_coding_question:
        return "solve_coding_question"
    else:
        return "solve_simple_question"

def solve_coding_question(state: State):
    user_message = state.get("user_message")

    SYSTEM_PROMPT = """
    You are an AI assistant. 
    Your job is to resolve the user's query based on coding problem he is facing
    Return the response in specified JSON boolean only.
    """

    result = client.beta.chat.completions.parse(
        model = "gemini-2.5-pro",
        response_format = CodingAIResponse, # AI will return response in this format (key: value)
        messages = [
            { "role": "system", "content": SYSTEM_PROMPT },
            { "role": "user", "content": user_message }
        ]
    )

    state["ai_answer"] = result.choices[0].message.parsed.answer
    return state

def solve_simple_question(state: State):
    user_message = state.get("user_message")

    SYSTEM_PROMPT = """
    You are an AI assistant. 
    Your job is to chat with user.
    """

    result = client.beta.chat.completions.parse(
        model = "gemini-2.5-flash-lite-preview-06-17",
        response_format = CodingAIResponse, # AI will return response in this format (key: value)
        messages = [
            { "role": "system", "content": SYSTEM_PROMPT },
            { "role": "user", "content": user_message }
        ]
    )

    state["ai_answer"] = result.choices[0].message.parsed.answer
    return state

# ---------------------- Build Graph ----------------------
graph_builder = StateGraph(State)

# Add all the nodes in graph builder (name of the node, actual node function)
graph_builder.add_node("detect_query", detect_query)
graph_builder.add_node("solve_coding_question", solve_coding_question)
graph_builder.add_node("solve_simple_question", solve_simple_question)
graph_builder.add_node("route_edge", route_edge)

# Flow
graph_builder.add_edge(START, "detect_query")
graph_builder.add_conditional_edges("detect_query", route_edge)

graph_builder.add_edge("route_edge", END)
graph_builder.add_edge("route_edge", END)

graph_builder.add_edge("solve_coding_question", END)
graph_builder.add_edge("solve_simple_question", END)

graph = graph_builder.compile() # it'll return a graph


# ---------------------- Use the Graph ----------------------
def call_graph():
    # define initial state
    state = {
        "user_message": "Hey there! How are you?",
        "ai_answer": "",
        "is_coding_question": False
    }

    result = graph.ainvoke(state) 
    print("Final Result: ", result)

call_graph()