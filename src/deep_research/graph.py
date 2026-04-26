"""LangGraph StateGraph wiring — compose nodes into a research pipeline."""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from deep_research.nodes.planner import planner_node
from deep_research.state import ResearchState


def build_graph() -> StateGraph:
    """Build and compile the research state graph.

    Current state (Day 3): planner only. More nodes added Day 4.
    """
    graph = StateGraph(ResearchState)
    graph.add_node("planner", planner_node)
    graph.add_edge(START, "planner")
    graph.add_edge("planner", END)
    return graph.compile(checkpointer=MemorySaver())


_graph = None


def get_graph() -> StateGraph:
    """Lazy singleton for the compiled graph."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
