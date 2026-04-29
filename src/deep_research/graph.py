"""LangGraph StateGraph wiring."""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from deep_research.nodes.critic import critic_node
from deep_research.nodes.planner import planner_node
from deep_research.nodes.searcher import searcher_node
from deep_research.nodes.synthesizer import synthesizer_node
from deep_research.state import ResearchState


def _route_after_critic(state: ResearchState) -> str:
    """Route to searcher for another iteration, or END."""
    feedback = state.get("critic_feedback", "done")
    iteration = state.get("iteration", 0)
    if feedback not in ("done", "max_iterations") and iteration < 2:
        return "searcher"
    return END


def build_graph() -> StateGraph:
    """Build the full research pipeline."""
    graph = StateGraph(ResearchState)
    graph.add_node("planner", planner_node)
    graph.add_node("searcher", searcher_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("critic", critic_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "searcher")
    graph.add_edge("searcher", "synthesizer")
    graph.add_edge("synthesizer", "critic")
    graph.add_conditional_edges(
        "critic",
        _route_after_critic,
        {"searcher": "searcher", END: END},
    )

    return graph.compile(checkpointer=MemorySaver())


_graph = None


def get_graph() -> StateGraph:
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
