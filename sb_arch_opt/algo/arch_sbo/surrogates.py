# --- 1) custom surrogate: GraphKernelKRG ------------------------------------
import numpy as np
from smt.surrogate_models import KRG
from smt.utils.kriging import cross_distances


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
        graph_kernel,
        create_graph=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.gp = graph_processor
        self.norm = normalization
        self.graph_kernel = graph_kernel
        self.create_graph = create_graph

        # DV -> graph cache
        self._dv_to_graph = {}

        # graph kernels have no gradients
        self.options["hyper_opt"] = "Cobyla"

    def _xnorm_to_raw_dv(self, x_row_norm):
        # verify this inverse in normalization class
        return self.norm.backward(np.asarray(x_row_norm).ravel().tolist())

    def _get_graph_for_xnorm(self, x_row_norm):
        raw_dv = self._xnorm_to_raw_dv(x_row_norm)

        # canonicalize + get corrected dv
        _g_none, dv_corr, _is_active = self.gp.get_graph(raw_dv, create=False)
        key = tuple(dv_corr)

        g = self._dv_to_graph.get(key)
        if g is not None:
            return g

        # create graph once
        g, dv_corr2, _is_active2 = self.gp.get_graph(dv_corr, create=self.create_graph)
        self._dv_to_graph[tuple(dv_corr2)] = g
        return g

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
        # SMT training X (what KRG uses to build R)
        X = self.training_points[None][0][0]

        _, ij = cross_distances(X)

        r = np.empty((ij.shape[0], 1), dtype=float)
        for k, (i, j) in enumerate(ij):
            Gi = self._get_graph_for_xnorm(X[i])
            Gj = self._get_graph_for_xnorm(X[j])
            r[k, 0] = float(self.graph_kernel(Gi, Gj))

        return np.clip(r, 1e-14, 1.0)
