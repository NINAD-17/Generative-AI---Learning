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

load_dotenv()

# Initialize LLM model
llm = init_chat_model("google_genai:gemini-2.0-flash")

# Initialize MemorySaver checkpointer
memory = MemorySaver()

# Initialize Tavily Search - a search engine for AI agents
search_web_tool = TavilySearch(max_results=2)

# Add all the tools
tools = [search_web_tool]

# Bind the tools to LLM so that it will have knowledge of all the available tools
llm_with_tools = llm.bind_tools(tools)

# State
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Nodes
# Node to run tools
tool_node = ToolNode(tools=[search_web_tool]) # REPLACED - BasicToolNode

# ChatBot Node
def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Route Edge
def route_tools(state: State):
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """

    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        print(f"Routing to tools: {ai_message.tool_calls}")
        return "tools"
    
    print("❌ No tool calls found, going to END")
    return END

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