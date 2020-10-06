"""Module that implements rendering utilities."""

import graphviz

from src.pdfa import PDFA


def to_graphviz(pdfa: PDFA) -> graphviz.Digraph:
    """Transform a PDFA to Graphviz."""
    graph = graphviz.Digraph(format="svg")
    graph.node("fake", style="invisible")

    for state in pdfa.states:
        if state == pdfa.initial_state:
            graph.node(str(state), root="true")
        else:
            graph.node(str(state))
    graph.node(str(pdfa.final_state), shape="doublecircle")

    graph.edge("fake", str(pdfa.initial_state), style="bold")

    for (start, char, prob, end) in pdfa.transitions:
        graph.edge(str(start), str(end), label=f"{char}, {prob}")

    return graph
