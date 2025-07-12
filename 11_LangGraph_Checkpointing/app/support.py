# ADMIN page
from .graph import create_chat_graph
import os
import json
from dotenv import load_dotenv
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.types import Command 

load_dotenv()

MONGODB_URI = os.getenv("MONGO_URI")
config = { "configurable": { "thread_id": "3" }} # config for unique user id

def init():
    # Create a checkpointer
    with MongoDBSaver.from_conn_string(MONGODB_URI) as checkpointer:
        graph_with_mongo = create_chat_graph(checkpointer = checkpointer)
    
        state = graph_with_mongo.get_state(config = config)
        
        # All messages history of thread ID
        for message in state.values["messages"]:
            message.pretty_print()

        print("----------------------------- History Restored ---------------------------\n")

        # Last Message
        last_message = state.values["messages"][-1]
        tool_calls = last_message.tool_calls

        print("last Msg: ", last_message)
        print("tool_calls", tool_calls)

        user_query = None

        for call in tool_calls:
            if call.get("name") == "human_assistance_tool":
                args_dict = call.get("args", {})
                user_query = args_dict.get("query")
                print(user_query)

        
        print("User is Tying to Ask:", user_query)
        support_ans = input("Resolution > ")
        format_support_ans = f"Human support replied: {support_ans};\nNow give this answer to the user"

        resume_command = Command(resume = { "data": format_support_ans }) # return data - in human_assitance_tool we're returning human_response["data"]

        for event in graph_with_mongo.stream(resume_command, config, stream_mode = "values"):
            if "messages" in event:
                event["messages"][-1].pretty_print()
init()