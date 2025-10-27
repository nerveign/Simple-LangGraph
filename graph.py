from langgraph.graph import StateGraph, START, END
from nodes import State, router, therapist_agent, logical_agent, universal_normalizer, message_classifier
from config import llm

def build_graph():
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

    return graph.compile()
