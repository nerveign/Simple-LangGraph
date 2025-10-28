from typing import Annotated, Literal
from pydantic import Field, BaseModel
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from config import llm

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
    return {"messages": [{"role":"assistant", "content": reply.content}]}

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
