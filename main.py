from typing import Annotated, Literal
from pydantic import Field,BaseModel
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o-mini", api_key=API_KEY)

# Definisi state
class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_type: str | None


class MessageClassifier(BaseModel):
    message_type: Literal["emotional", "logical"] = Field(
        ...,
        description="Classify if the message requires an emotional (therapist) or logical response."
    )

# Node: get user input
def get_input_user(state: State):
    user_input = input("Enter your message: ")
    return {"messages": [{"role": "user", "content": user_input}]}

# Node: universal normalizer
def universal_normalizer(state: State):
    last_message = state["messages"][-1].content.lower().strip()
    return {"messages": [{"role": "assistant", "content": last_message}]}

def message_classifier(state: State):
    last_message = state["messages"][-1].content
    classifier_llm = llm.with_structured_output(MessageClassifier)

    result = classifier_llm.invoke([
        {
            "role": "system",
            "content": """Classify the user message as either:
            - 'emotional': if it asks for emotional support, therapy, deals with feelings, or personal problems
            - 'logical': if it asks for facts, information, logical analysis, or practical solutions
            """
        },
        {"role": "user", "content": last_message}
    ])
    return {"message_type": result.message_type}

# Create Graph
graph = StateGraph(State)

graph.add_node("get_input_user", get_input_user)
graph.add_node("universal_normalizer", universal_normalizer)
graph.add_node("message_classifier", message_classifier)

graph.add_edge(START, "get_input_user")
graph.add_edge("get_input_user", "universal_normalizer")
graph.add_edge("universal_normalizer", "message_classifier")
graph.add_edge("message_classifier", END)

# Compile Graph
app = graph.compile()

# Run Graph
final_state = app.invoke({"messages": []})

print("Normalize result: ", final_state["messages"][-1].content)
print("Message type: ", final_state["message_type"])

