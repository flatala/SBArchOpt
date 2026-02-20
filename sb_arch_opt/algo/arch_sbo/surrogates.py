# --- 1) custom surrogate: GraphKernelKRG ------------------------------------
import networkx as nx
import numpy as np
from smt.surrogate_models import KRG
from smt.utils.kriging import cross_distances
import grakel as gk
from grakel.kernels import WeisfeilerLehman, VertexHistogram

from adore.graph import FunctionNode
from adsg_core import DSGType


# Maybe it would make sense to just integrate the output o a kernl into the design vector???
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
        graph_processor,
        normalization,
        graph_kernel=None, # For now we dont use it / should be
        **kwargs,
    ):
        super().__init__(**kwargs)
        print("GraphKernelKRG init id:", id(self))
        self.gp = graph_processor
        self.norm = normalization
        self.graph_kernel = graph_kernel

        # DV -> graph cache
        self._dv_to_graph = {}

        # graph kernels have no gradients
        self.options["hyper_opt"] = "Cobyla"

        # TODO - This could be in principle some mapping object that is passed to the odel or sth, not a dict
        #  / think abt some kernel data extractor + computation interface
        # self._node_type_label_dict = {
        #
        # }

    # def _xnorm_to_raw_dv(self, x_row_norm):
    #
    #     ds = self.norm._design_space
    #
    #     print("design_space object:", ds)
    #     print("design_space type:", type(ds))
    #     print("xl type:", type(ds.xl), "xu type:", type(ds.xu))
    #
    #     import inspect
    #     print("xl defined on class as:", type(getattr(type(ds), "xl", None)))
    #
    #     return self.norm.backward(np.asarray(x_row_norm, dtype=float))

    def _xnorm_to_raw_dv(self, x_row_norm):
        x2d = np.asarray(x_row_norm, dtype=float).ravel()[None, :]  # shape (1, n_var)
        raw2d = self.norm.backward(x2d)  # shape (1, n_var)
        return raw2d[0]

    def _get_graph_for_xnorm(self, x_row_norm):
        raw_dv = self._xnorm_to_raw_dv(x_row_norm)

        # First without creating the graph to ony get the corrected
        # dv and check if we have a cache hit.
        _, dv_corr, active = self.gp.get_graph(raw_dv, create=False)
        key = tuple(dv_corr)

        # check for cache hit
        g = self._dv_to_graph.get(key)
        if g is not None:
            return g

        # create graph once
        g , _, _ = self.gp.get_graph(dv_corr, create=True)

        # nxMultiDiGraph
        G = g.graph

        # TODO: make sure this gets collapsed right (ie if we have two parallel edges we egt a 2 in the adjacency)
        nodes = list(G.nodes())
        # edges = G.edges - unused for now
        A = nx.to_numpy_array(G, nodelist=nodes, weight="weight")
        node_labels = {i: 0 for i in range(len(nodes))}

        # TODO - implement this properly, using fixed label for all nodes for now
        # we need ot assign labels, for now let's do it by adsg node types
        # function node -> comp -> etc
        # The idea is to assign a higher integer to a higher level node type
        # node_labels = []
        # for node in nodes:
        #     if node is FunctionNode:
        #         label = self._get_node_label(node)
        #         node_labels.append(label)

        # TODO - we also might want edge labels for some kernels. Maybe we could provide some extractor
        #  objects with a kernel to combine extraction and kernel calculation
        # edge_labels = []

        # build and cache the grakel graph
        gk_graph = gk.Graph(A, node_labels=node_labels)
        self._dv_to_graph[tuple(dv_corr)] = gk_graph
        return gk_graph

    def _get_node_label(self, node):
        cls = type(node)
        try:
            return self._node_type_label_dict[cls]
        except KeyError as e:
            raise KeyError(f"No label registered for {cls.__module__}.{cls.__qualname__}") from e

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
        raise RuntimeError("_matrix_data_corr called")
        X = self.training_points[None][0][0]

        _, ij = cross_distances(X)

        graphs = [self._get_graph_for_xnorm(X[i]) for i in range(X.shape[0])]

        # calculate kernel
        wl = WeisfeilerLehman(n_iter=3, base_graph_kernel=VertexHistogram)
        K = wl.fit_transform(graphs)

        diag = np.clip(np.diag(K), 1e-30, None)
        inv_sqrt_d = 1.0 / np.sqrt(diag)

        # Gram matrix needs to be PSD, so we normalize as K_norm = D^{-1/2} K D^{-1/2}, this preserves PSD
        K_norm = (K * inv_sqrt_d[None, :]) * inv_sqrt_d[:, None]
        K_norm = 0.5 * (K_norm + K_norm.T)
        r = np.empty((ij.shape[0], 1), dtype=float)
        for k, (i, j) in enumerate(ij):
            r[k, 0] = float(K_norm[i, j])

        return r
