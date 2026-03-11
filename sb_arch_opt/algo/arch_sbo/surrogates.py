from smt.surrogate_models import KRG
from smt.utils.kriging import cross_distances
from grakel.kernels import WeisfeilerLehman, VertexHistogram
from adsg_core import DSGType
from adsg_core.optimization.graph_processor import GraphProcessor
from pymoo.util.normalization import Normalization

import grakel as gk
import networkx as nx
import numpy as np

from abc import ABC, abstractmethod
from typing import Any, Sequence

class GraphKernelBuilder(ABC):
    @abstractmethod
    def build_graph(self, G: DSGType) -> Any:
        """Create the graph object used by the kernel (e.g., a GraKeL graph)."""

    @abstractmethod
    def fit_transform(self, train_graphs: Sequence[Any]) -> Any:
        """Return K_train_train (n_train x n_train) and store fitted state internally."""

    @abstractmethod
    def transform(self, test_graphs: Sequence[Any]) -> Any:
        """Return K_test_train (n_test x n_train) using stored fitted state."""

class GraphKernelKRG(KRG):
    """
    KRG where correlation is computed via a graph kernel on graphs built from design vectors.
    Caching is inside the surrogate.

    Requirements:
    - graph_processor: has get_graph(dv, create=...) -> (graph, dv_corr, is_active) or similar
    - normalization: maps between raw DV space and SMT training X space
        - forward(raw_dv)  -> x_norm  (already used elsewhere)
        - backward(x_norm) -> raw_dv  (you must provide this inverse)
    - graph_kernel(Gi, Gj) -> float in (0,1] (or at least >0)
    """

    def __init__(
        self,
        graph_processor: GraphProcessor,
        normalization: Normalization,
        kernel_builder: GraphKernelBuilder,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.gp = graph_processor
        self.norm = normalization
        self.kernel_builder = kernel_builder
        self._raw_to_corr: dict[tuple, tuple] = {}
        self._corr_to_graph: dict[tuple, Any] = {}
        self.K_train_train = None
        self.options["hyper_opt"] = "Cobyla"

    def _x_norm_to_raw_dv(self, x_row_norm):
        x2d = np.asarray(x_row_norm, dtype=float).ravel()[None, :]  # shape (1, n_var)
        raw2d = self.norm.backward(x2d)  # shape (1, n_var)
        return raw2d[0]

    def _get_graph_for_x_norm(self, x_row_norm) -> Any:
        # Unnormalize the design vector
        raw_dv = self._x_norm_to_raw_dv(x_row_norm)

        # Check if this raw_dv was encountered
        raw_key = tuple(raw_dv)
        dv_corr = self._raw_to_corr.get(raw_key)

        # If raw_dv was not encountered, correct the dv
        # Also add the raw -> corr cache entry
        if dv_corr is None:
            _, dv_corr_arr, _ = self.gp.get_graph(raw_dv, create=False)
            dv_corr = tuple(dv_corr_arr)
            self._raw_to_corr[raw_key] = dv_corr

        # Check if a graph for the corrected dv was already created
        built_graph = self._corr_to_graph.get(dv_corr)
        if built_graph is not None:
            return built_graph

        # Create the graph
        g, _, _ = self.gp.get_graph(np.asarray(dv_corr, dtype=float), create=True)
        built_graph = self.kernel_builder.build_graph(g)

        # Cache the resulting graph
        self._corr_to_graph[dv_corr] = built_graph

        return built_graph

    def _new_train(self):
        # Reset the matrix and ij so the model gets fitted again
        self._ij_train = None
        self.K_train_train = None
        super()._new_train()

    def _matrix_data_corr(
        self,
        corr,
        design_space,
        power,
        theta,
        theta_bounds,
        dx,
        Lij=None,
        n_levels=None,
        cat_features=None,
        cat_kernel=None,
        x=None,
        kplsk_second_loop=False,
    ):
        X_train = self.training_points[None][0][0]

        if self.K_train_train is None:
            _, self._ij_train = cross_distances(X_train)
            n_train = X_train.shape[0]
            # Get built graph instances
            train_graphs = [self._get_graph_for_x_norm(X_train[i]) for i in range(n_train)]
            K = self.kernel_builder.fit_transform(train_graphs)
            self.K_train_train = K

        if x is None:
            # Array of (i, j) pairs of len n * (n - 1) / 2
            ij = self._ij_train
            r = np.empty((ij.shape[0], 1), dtype=float)
            # Put the values form the matrix in an array
            for k, (i, j) in enumerate(ij):
                r[k, 0] = float(self.K_train_train[i, j])
            return r

        # Based on fitted kernel on train points, we get the test-train point covariances K(X*, X)
        x = np.asarray(x, dtype=float)
        n_eval = x.shape[0]
        test_graphs = [self._get_graph_for_x_norm(x[i]) for i in range(n_eval)]
        K_x_train = self.kernel_builder.transform(test_graphs)
        assert K_x_train.shape == (n_eval, X_train.shape[0])
        return K_x_train.reshape(-1, 1)

class SimpleWLKernelBuilder(GraphKernelBuilder):
    def __init__(self):
        self.kernel = WeisfeilerLehman(n_iter=3, base_graph_kernel=VertexHistogram, normalize=True)

    def build_graph(self, G: DSGType) -> Any:
        # noinspection PyTypeChecker
        G_nx: nx.MultiDiGraph = G.graph
        nodes = list(G_nx.nodes())
        A = nx.to_numpy_array(G_nx, nodelist=nodes, weight="weight")
        node_labels = {i: 0 for i in range(len(nodes))}
        gk_graph = gk.Graph(A, node_labels=node_labels)
        return gk_graph

    def fit_transform(self, train_graphs: Sequence[Any]) -> Any:
        K_train_train = self.kernel.fit_transform(train_graphs)
        return K_train_train

    def transform(self, test_graphs: Sequence[Any]) -> Any:
        K_x_train = self.kernel.transform(test_graphs)
        return K_x_train

# TODO: finish up, make teh assignment to integers make sense in terms of the hierarchy between nodes
class WLVHKernelBuilder(GraphKernelBuilder):
    def __init__(self):
        self.kernel = WeisfeilerLehman(n_iter=3, base_graph_kernel=VertexHistogram, normalize=True)
        self._class_name_to_label = {name: i for i, name in enumerate(['FunctionNode', 'ComponentNode',
            'ComponentInstanceNode', 'SystemNode', 'ComponentInstanceGroupNode', 'GroupNode', 'PortGroupNode',
            'NopNode', 'ConnectorDegreeGroupingNode', 'ConnectorNode', 'AttributeNode', 'AttributeValueNode',
            'InputParamNode', 'DesignVariableNode', 'MetricNode', 'FunctionDerivationNode', 'ConceptNode',
            'FunctionDecompositionNode', 'ExternalConnectionNode', 'ExternalOutConnectionNode', 'SystemGroupNode',
            'NonFulfillmentNode', 'MultiFulfillmentNode'])}

    def build_graph(self, G: DSGType) -> Any:
        # noinspection PyTypeChecker
        G_nx: nx.MultiDiGraph = G.graph
        nodes = list(G_nx.nodes())
        A = nx.to_numpy_array(G_nx, nodelist=nodes, weight="weight")
        node_labels = {i: 0 for i in range(len(nodes))}
        gk_graph = gk.Graph(A, node_labels=node_labels)
        return gk_graph

    def _get_node_label(self, node: Any) -> int:
        cls_name = node.__class__.__name__
        try:
            return self._class_name_to_label[cls_name]
        except KeyError as e:
            raise ValueError(f"Unknown node class '{cls_name}' (not in __all__)") from e


    def fit_transform(self, train_graphs: Sequence[Any]) -> Any:
        K_train_train = self.kernel.fit_transform(train_graphs)
        return K_train_train

    def transform(self, test_graphs: Sequence[Any]) -> Any:
        K_x_train = self.kernel.transform(test_graphs)
        return K_x_train