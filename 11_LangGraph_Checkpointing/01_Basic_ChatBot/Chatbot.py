import json
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from IPython.display import Image, display
from langchain_tavily import TavilySearch
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt
from langchain_core.tools import tool

load_dotenv()

# Initialize LLM model
llm = init_chat_model("google_genai:gemini-2.0-flash")

# Initialize MemorySaver checkpointer
memory = MemorySaver()

# Initialize Tavily Search Tool - a search engine for AI agents
search_web_tool = TavilySearch(max_results=2)

# Human Assitance Tool
@tool # It transform a standard Python function into a LangChain-compatible tool object
def human_assistance(query: str) -> str:
    """Request assistance from a human."""
    human_response = interrupt({"query": query})
    return human_response["data"]

# Add all the tools
tools = [search_web_tool, human_assistance]

# Bind the tools to LLM so that it will have knowledge of all the available tools
llm_with_tools = llm.bind_tools(tools)

# State
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Nodes
# Node to run tools
tool_node = ToolNode(tools = tools) # REPLACED - BasicToolNode

# ChatBot Node
def chatbot(state: State):
    message = llm_with_tools.invoke(state["messages"])
    
    assert len(message.tool_calls) <= 1 # to ensure that only one tool is called because we're using interrupt which will stop the execution
    return {"messages": [message]}

# Create StateGraph object to define Graph structure
graph_builder = StateGraph(State)

# Define Nodes
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)

# Conditional Edges
graph_builder.add_conditional_edges(
    "chatbot", 
    tools_condition, # REPLACED - route_tools
    {
        "tools": "tools",  # When route_tools() returns "tools", go to the node called "tools"
        END: END           # When it returns END, stop the graph.
    }
)

# Edges
graph_builder.add_edge(START, "chatbot") # user input -> chatbot
graph_builder.add_edge("tools", "chatbot") # if llm asked to call the tool, route_tools -> get output from the tool -> append the output to messages -> chatbot (to get answer from llm with the appended tool output)
# graph_builder.add_edge("chatbot", END)

# Compile to create a graph from the structure that we've defined
graph = graph_builder.compile(checkpointer = memory)

# Display the Graph (Run it in Jupyter Notebook)
try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass

# Run the graph
def stream_graph_updates(user_input: str, config: dict):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}, config, stream_mode="values"):
        event["messages"][-1].pretty_print()

while True:
    # Checkpointer Thread Id (To recognize individual conversation). You can try it by changing the thread_id
    config = {"configurable": {"thread_id": "1"}} 

    try:
        snapshot = graph.get_state(config)
        if snapshot.next == ("tools",):
            print("🔔 Execution paused. Awaiting human input...")
            data = input("Human input: ")
            command = Command(resume={"data": data})
            for event in graph.stream(command, config, stream_mode="values"):
                event["messages"][-1].pretty_print()
                
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(user_input, config)
    except:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input, config)
        break