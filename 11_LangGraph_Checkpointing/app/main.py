# from .graph import graph
from .graph import create_chat_graph
import os
from dotenv import load_dotenv
from langgraph.checkpoint.mongodb import MongoDBSaver

load_dotenv()

MONGODB_URI = os.getenv("MONGO_URI")
config = { "configurable": { "thread_id": "3" }} # config for unique user id

def init():
    # Create a checkpointer
    with MongoDBSaver.from_conn_string(MONGODB_URI) as checkpointer:
        graph_with_mongo = create_chat_graph(checkpointer = checkpointer)

        # All messages history of thread ID
        state = graph_with_mongo.get_state(config = config)
        if state and state.values and state.values["messages"]:
            for message in state.values["messages"]:
                message.pretty_print()
            print("--------------------- Chat Restored -----------------------\n")
        else:
            print("----------------------- New Chat ----------------------------\n")
    
        while True:
            user_input = input("You > ")

            # graph.invoke() will return result at last
            # graph.stream() will return event at each node
            # result = graph.invoke({ "messages": [{ "role": "user", "content": user_input }] 
            # for event in graph.stream({ "messages": [{ "role": "user", "content": user_input }]}, stream_mode = "values"):
                
            for event in graph_with_mongo.stream({ "messages": [{ "role": "user", "content": user_input }]}, config, stream_mode = "values"):
                if "messages" in event: 
                    event["messages"][-1].pretty_print()

init()