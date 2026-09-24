"""Domain-independent graph representations used by graph surrogates."""

from dataclasses import dataclass, field
from typing import Any, Callable, Hashable, Literal, Mapping, Protocol, Sequence, runtime_checkable

import networkx as nx
import numpy as np


__all__ = ["NodeLabeling", "SizingFeature", "GraphRepresentation", "GraphDecoder"]


@dataclass(frozen=True)
class NodeLabeling:
    """One named way of labeling nodes."""

    key: str
    function: Callable[[Any], Hashable] = field(compare=False, hash=False, repr=False)

    def __call__(self, node: Any) -> Hashable:
        return self.function(node)


@dataclass(frozen=True)
class SizingFeature:
    """One normalized sizing coordinate exported by a graph decoder."""

    name: str
    kind: Literal["numeric", "categorical"]


def _empty_float_array() -> np.ndarray:
    return np.empty((0,), dtype=float)


@dataclass(frozen=True)
class GraphRepresentation:
    """Neutral graph, node labels, and sizing values used by graph kernels."""

    graph: nx.Graph
    node_labels: Mapping[NodeLabeling, Mapping[Hashable, Hashable]]
    sizing_values: np.ndarray = field(default_factory=_empty_float_array)


@runtime_checkable
class GraphDecoder(Protocol):
    """Converts design vectors into neutral graph representations."""

    @property
    def labelings(self) -> Sequence[NodeLabeling]:
        ...

    @property
    def sizing_features(self) -> Sequence[SizingFeature]:
        ...

    def decode(self, x: np.ndarray) -> Sequence[GraphRepresentation]:
        ...
