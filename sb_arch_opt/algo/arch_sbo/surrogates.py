# --- 1) custom surrogate: GraphKernelKRG ------------------------------------
import networkx as nx
import numpy as np
from smt.surrogate_models import KRG
from smt.utils.kriging import cross_distances
import grakel as gk
from grakel.kernels import WeisfeilerLehman, VertexHistogram

from adore.graph import FunctionNode
from adsg_core import DSGType

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

    # def _get_node_label(self, node):
    #     cls = type(node)
    #     try:
    #         return self._node_type_label_dict[cls]
    #     except KeyError as e:
    #         raise KeyError(f"No label registered for {cls.__module__}.{cls.__qualname__}") from e

    def _matrix_data_corr(
        self,
        corr, # type of correlation model
        design_space,
        power,
        theta, # hyperpaams of teh correlation model
        theta_bounds,
        dx, # tensor pf gower componentwise distances betweens samples
        Lij=None,
        n_levels=None, # levels for every cat variable
        cat_features=None, # indices of cat variables
        cat_kernel=None,
        x=None, # in[ut instead of dx for homo_hs prediction???
        kplsk_second_loop=False,
    ):
        X_train = self.training_points[None][0][0]
        n_train = X_train.shape[0]

        # get networkX graphs
        train_graphs = [self._get_graph_for_xnorm(X_train[i]) for i in range(n_train)]

        # array of (i, j) pairs of len n * (n - 1) / 2
        _, ij_train = cross_distances(X_train)

        # fit and calculate train-tain kernel K(X, X) - this is super inefficient for now, wi double fit it but whatever for now
        wl = WeisfeilerLehman(n_iter=3, base_graph_kernel=VertexHistogram, normalize=True)
        K_train_train = wl.fit_transform(train_graphs)

        if x is None:
            ij = ij_train
            K = K_train_train
            r = np.empty((ij.shape[0], 1), dtype=float)

            # put the values form the matrix in an array
            for k, (i, j) in enumerate(ij):
                r[k, 0] = float(K[i, j])

            return r

        x = np.asarray(x, dtype=float)
        n_eval = x.shape[0]

        # based on fitted kernel on train points, get the test-train point covariances K(X*, X)
        test_graphs = [self._get_graph_for_xnorm(x[i]) for i in range(n_eval)]
        K_x_train = wl.transform(test_graphs)

        return K_x_train.reshape(-1, 1)