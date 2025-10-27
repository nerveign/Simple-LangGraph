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

class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_type: str | None

class MessageClassifier(BaseModel):
    message_type: Literal["emotional", "logical"] = Field(
        ...,
        description="Classify if the message requires an emotional (therapist) or logical response."
    )

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

def router(state: State):
    message_type = state.get("message_type", "logical")
    if message_type == "emotional":
        return {"next": "therapist"}

    return {"next": "logical"}

def therapist_agent(state: State):
    last_message = state["messages"][-1]

    messages = [
        {
            "role": "system",
            "content": """You are compassionate therapist. Focus on the emotional aspects of the user's message.
                Show empathy, validate their feelings, and help them process their emotions.
                Ask thoughtful questions to help them explore their feelings more deeply.
                Avoid giving logical solutions unless explicitly asked."""
        },
        {
            "role": "user",
            "content": last_message.content
        }
    ]
    
    reply = llm.invoke(messages)

    return {
        "messages": [
            {
            "role":"assistant",
            "content": reply.content
            }
        ]
    }


def logical_agent(state: State):
    last_message = state["messages"][-1]

    messages = [
        {
            "role": "system",
            "content": """You are a purely logical assistant. Focus only on facts and information.
            Provide clear, concise answers based on the information provided.
            Avoid giving emotional support or personal opinions."""
        },
        {
            "role": "user",
            "content": last_message.content
        }
    ]

    reply = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": reply.content}]}

# Create Graph
graph = StateGraph(State)

graph.add_node("universal_normalizer", universal_normalizer)
graph.add_node("message_classifier", message_classifier)
graph.add_node("router", router)
graph.add_node("therapist_agent", therapist_agent)
graph.add_node("logical_agent", logical_agent)

graph.add_edge(START, "universal_normalizer")
graph.add_edge("universal_normalizer", "message_classifier")
graph.add_edge("message_classifier", "router")
graph.add_conditional_edges(
    "router",
    lambda state: state.get("next"),
    {"therapist": "therapist_agent", "logical": "logical_agent"}
)
graph.add_edge("therapist_agent", END)
graph.add_edge("logical_agent", END)

# Compile Graph
app = graph.compile()

def run_chatbot():
    state = {"messages": [], "message_type": None}

    while True:
        user_input = input("Message: ")
        if user_input == "exit":
            print("Bye")
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