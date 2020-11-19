"""Module that implements rendering utilities."""
from typing import Callable, Dict, Set

import graphviz

from src.pdfa import PDFA
from src.pdfa.helpers import (
    PROB_LOWER_BOUND,
    ROUND_PRECISION,
    filter_transition_function,
)
from src.pdfa.types import Character


def to_graphviz(
    pdfa: PDFA,
    state2str: Callable[[int], str] = lambda x: str(x),
    char2str: Callable[[int], str] = lambda x: str(x),
    round_precision: int = ROUND_PRECISION,
    lower_bound: float = PROB_LOWER_BOUND,
) -> graphviz.Digraph:
    """Transform a PDFA to Graphviz."""
    graph = graphviz.Digraph(format="svg")
    graph.node("fake", style="invisible")

    states, filtered_transition_function = filter_transition_function(
        pdfa.transition_dict, lower_bound
    )

    for state in states:
        if state == pdfa.initial_state:
            graph.node(state2str(state), root="true")
        else:
            graph.node(state2str(state))
    graph.node(state2str(pdfa.final_state), shape="doublecircle")

    graph.edge("fake", state2str(pdfa.initial_state), style="bold")

    for start, outgoing in filtered_transition_function.items():
        for char, (end, prob) in outgoing.items():
            new_prob = round(prob, round_precision)
            if new_prob > lower_bound:
                graph.edge(
                    state2str(start),
                    state2str(end),
                    label=f"{char2str(char)}, {new_prob}",
                )

    return graph


# TODo refactor
def to_graphviz_from_graph(
    vertices: Set[int],
    transitions: Dict[int, Dict[Character, int]],
    state2str: Callable[[int], str] = lambda x: str(x),
    char2str: Callable[[int], str] = lambda x: str(x),
):
    """To graphviz from graph."""
    graph = graphviz.Digraph(format="svg")
    graph.node("fake", style="invisible")

    for state in vertices:
        if state == 0:
            graph.node(state2str(state), root="true")
        else:
            graph.node(state2str(state))

    graph.edge("fake", state2str(0), style="bold")

    for start, char2end in transitions.items():
        for char, end in char2end.items():
            graph.edge(
                state2str(start),
                state2str(end),
                label=f"{char2str(char)}",
            )

    return graph
