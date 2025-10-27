from graph import build_graph

# Compile Graph
app = build_graph()

def run_chatbot():
    state = {"messages": [], "message_type": None}

    while True:
        user_input = input("Message: ")
        if user_input == "exit":
            print("Thanks for using our service, bye!")
            break
        
        state["messages"] = state.get("messages", []) + [
            {"role": "user", "content": user_input}
        ]

        state = app.invoke(state)
        
        if state.get("messages") and len(state["messages"]) > 0:
            last_message = state["messages"][-1]
            print(f"Assistant: {last_message.content}")

if __name__ == "__main__":
    run_chatbot()