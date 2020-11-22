"""Interface and implementation of a multiset."""
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from typing import (
    Collection,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from src.types import Character, Word


@dataclass
class _TreeMetadata:
    """Keep tree data."""

    size: int = 0
    alphabet_size: int = 0

    def __hash__(self):
        return id(self)


def prefixes(t: Word) -> Iterable:
    """Return all the prefixes of a trace."""
    # len + 1, so it is guaranteed to do
    # at least one iteration (in case of empty trace)
    for i in range(len(t) + 1):
        yield t[:i]


class Node:
    """A node in the prefix-tree."""

    __slots__ = [
        "_index",
        "_tree_metadata",
        "_parent",
        "_symbol",
        "_symbol2child",
        "counts",
        "children_counts",
    ]

    def __init__(self, parent: Optional["Node"], symbol: Optional[int] = None):
        """Initialize the prefix-tree node."""
        self._parent = parent
        self._symbol = symbol
        self.counts = 0
        self.children_counts = 0
        self._symbol2child: Dict[int, Node] = {}
        if parent is not None:
            assert symbol is not None
            self._tree_metadata: _TreeMetadata = parent._tree_metadata
            self._index = self._tree_metadata.size
            parent.add_child(symbol, self)
        else:
            self._tree_metadata = _TreeMetadata(size=1)
            self._index = 0

    def add_child(self, symbol: int, node: "Node") -> None:
        """Add child."""
        assert symbol not in self._symbol2child
        self._tree_metadata.size += 1
        self._symbol2child[symbol] = node

    @property
    def index(self) -> int:
        """Get the index of the node."""
        return self._index

    def add(self, trace: Word) -> None:
        """Add a trace to the prefix tree."""
        current_node: Node = self
        current_node.children_counts += 1
        for character in trace:
            next_node = current_node._symbol2child.get(character, None)
            if next_node is None:
                # create a new node.
                next_node = Node(current_node, character)
            current_node = next_node
            current_node.children_counts += 1
        current_node.counts += 1

    def get_end_node(self, trace: Word) -> Optional["Node"]:
        """Get the finale node (after processing the entire trace)."""
        result: Optional[Node] = self
        for character in trace:
            if result is not None:
                result = result._symbol2child.get(character, None)
            if result is None:
                break
        return result

    def next_nodes(self) -> Set["Node"]:
        """Get the next nodes."""
        return set(self._symbol2child.values())

    def next_transitions(self) -> Collection[Tuple[Character, "Node"]]:
        """Get the next transitions."""
        return list(self._symbol2child.items())

    def traces(self) -> Set[Word]:
        """Get all traces from this node."""
        result: Set[Word] = set()

        prefix = (self._symbol,) if self._symbol is not None else ()
        if self.counts > 0:
            result.add(prefix)

        next_nodes = self.next_nodes()
        for node in next_nodes:
            next_traces = node.traces()
            if len(next_traces) > 0:
                new_traces = set(map(lambda x: prefix + tuple(x), next_traces))
                result = result.union(new_traces)
        return result

    def items(self) -> Set[Tuple[Word, int]]:
        """Get list of pairs, trace and its count."""
        result: Set[Tuple[Word, int]] = set()

        prefix = (self._symbol,) if self._symbol is not None else ()
        if self.counts > 0:
            result.add((prefix, self.counts))

        next_nodes = self.next_nodes()
        for node in next_nodes:
            next_traces_counts = node.items()
            if len(next_traces_counts) > 0:
                new_traces_counts = set(
                    map(lambda x: (prefix + tuple(x[0]), x[1]), next_traces_counts)
                )
                result = result.union(new_traces_counts)
        return result

    def get_counts(self, t: Word) -> int:
        """Get the counts of a trace."""
        if len(t) == 0:
            return self.counts
        for index, character in enumerate(t):
            next_node = self._symbol2child.get(character, None)
            if next_node is not None:
                # found next node.
                return next_node.get_counts(t[index + 1 :])
        # no next node found => trace is not in the multiset.
        return 0

    def __eq__(self, other: object) -> bool:
        """Check equality."""
        if not isinstance(other, Node):
            return NotImplemented
        return self.index == other.index

    def __hash__(self) -> int:
        """Get hash."""
        return hash((Node, self.index, self._tree_metadata))


class Multiset(ABC):
    """Abstract multiset."""

    @abstractmethod
    def get_counts(self, trace: Word) -> int:
        """Get counts."""

    @abstractmethod
    def add(self, t: Word, times: int = 1) -> None:
        """
        Add a trace in the multiset.

        :param t: the trace to add.
        :param times: how many times it should be added.
        :return: None
        """

    @property
    @abstractmethod
    def size(self) -> int:
        """Get the size."""

    @abstractmethod
    def get_probability(self, t: Word) -> float:
        """Get the probability of a trace."""

    @abstractmethod
    def get_prefix_probability(self, t: Word) -> float:
        """Get the prefix-probability of a trace."""

    @property
    @abstractmethod
    def traces(self) -> Set[Word]:
        """Get the traces."""

    def elements(self) -> Iterator[Word]:
        """Get the set of traces."""
        for trace, count in self.items():
            for _ in range(count):
                yield trace

    @abstractmethod
    def items(self) -> Set[Tuple[Word, int]]:
        """Get list of tuples (trace, count)."""

    def update(self, sample: Sequence[Word]):
        """Add items."""
        for t in sample:
            self.add(t)

    def __len__(self) -> int:
        """Get the length."""
        return self.size

    def __iter__(self):
        """Get the traces."""
        return iter(self.traces)

    def __getitem__(self, item):
        """Get the count."""
        return self.get_counts(item)

    def values(self) -> Sequence[int]:
        """Get the values."""
        return [v for _, v in self.items()]


class NaiveMultiset(Multiset):
    """Implement a multiset in a naive way - using a counter."""

    def __init__(self):
        """Initialize the multiset."""
        self._counter = Counter()

    def get_counts(self, trace: Word) -> int:
        """Get counts."""
        return self._counter[trace]

    def add(self, t: Word, times: int = 1) -> None:
        """Add an item to the multiset."""
        self._counter.update({t: times})

    @property
    def size(self) -> int:
        """Get the size of the multiset."""
        return sum(self._counter.values())

    def get_probability(self, t: Word) -> float:
        """Get the probability of a trace."""
        if self.size == 0:
            return 0
        return self._counter[t] / self.size

    def get_prefix_probability(self, t: Word) -> float:
        """Get the prefix-probability of a trace."""
        if self.size == 0:
            return 0
        p = 0.0
        for string in self._counter.keys():
            for i in range(len(string) + 1):
                prefix, suffix = string[:i], string[i:]
                if prefix != t:
                    continue
                p += self._counter[prefix + suffix]

        return p / self.size

    @property
    def traces(self) -> Set[Word]:
        """Get the set of traces."""
        return set(map(tuple, self._counter.keys()))

    def items(self) -> Set[Tuple[Word, int]]:
        """Get the traces and their counts."""
        return set(self._counter.items())


class PrefixTreeMultiset(Multiset):
    """A multi-set based on a prefix tree."""

    def __init__(self, node: Optional[Node] = None):
        """
        Initialize a Multiset prefix-tree based.

        :param node: the node of the tree from where to start.
        """
        self._node = node if node is not None else Node(parent=None)

    def get_counts(self, trace: Word) -> int:
        """Get counts."""
        return self._node.get_counts(trace)

    def add(self, t: Word, times: int = 1) -> None:
        """Add an element."""
        self._node.add(t)

    @property
    def size(self) -> int:
        """Get the size."""
        return self._node.children_counts

    def get_probability(self, t: Word) -> float:
        """Get the probability of a trace."""
        if self._node.children_counts == 0:
            return 0.0
        return self._node.get_counts(t) / self.size

    def get_prefix_probability(self, t: Word) -> float:
        """Get the prefix-probability of a trace."""
        if self._node.children_counts == 0:
            return 0.0
        final_node: Optional[Node] = self._node.get_end_node(t)
        if final_node is None:
            # never seen this prefix.
            return 0.0
        return final_node.children_counts / self.size

    @property
    def traces(self) -> Set[Word]:
        """Get the set of traces."""
        return self._node.traces()

    def items(self) -> Set[Tuple[Word, int]]:
        """Get the traces and their counts."""
        return self._node.items()


class ReadOnlyPrefixTreeMultiset(Multiset):
    """Readonly multiset."""

    def __init__(self, nodes: Optional[Set[Node]] = None):
        """
        Initialize a Multiset prefix-tree based.

        :param nodes: the nodes of the tree from where to start.
        """
        self._nodes = nodes if nodes is not None else {Node(parent=None)}

    def get_counts(self, trace: Word) -> int:
        """Get counts."""
        return sum(n.get_counts(trace) for n in self._nodes)

    def add(self, t: Word, times: int = 1) -> None:
        """Add an element."""
        raise ValueError("Read-only.")

    @property
    def size(self) -> int:
        """Get the size."""
        return sum(n.children_counts for n in self._nodes)

    def get_probability(self, t: Word) -> float:
        """Get the probability of a trace."""
        probabilities = [
            n.get_counts(t) / n.children_counts
            for n in self._nodes
            if n.children_counts != 0
        ]
        return sum(probabilities)

    def get_prefix_probability(self, t: Word) -> float:
        """Get the prefix-probability of a trace."""
        final_nodes: List[Optional[Node]] = [n.get_end_node(t) for n in self._nodes]
        return sum(
            final_node.children_counts / self.size
            for final_node in final_nodes
            if final_node is not None and final_node.children_counts > 0
        )

    @property
    def traces(self) -> Set[Word]:
        """Get the set of traces."""
        return set.union(*[n.traces() for n in self._nodes])

    def items(self) -> Set[Tuple[Word, int]]:
        """Get the traces and their counts."""
        return set.union(*[n.items() for n in self._nodes])
