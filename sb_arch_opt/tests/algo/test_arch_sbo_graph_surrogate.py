import networkx as nx
import numpy as np

try:
    from smt.design_space import DesignSpace, IntegerVariable
except ImportError:
    from smt.utils.design_space import DesignSpace, IntegerVariable

from sb_arch_opt.algo.arch_sbo.graph import (
    CompositeGraphKernel,
    GraphKriging,
    GraphRepresentation,
    LdWloa,
    NodeLabeling,
    SizingFeature,
    SizingKernel,
)


LABELING = NodeLabeling("kind", lambda node: node)


class ExampleDecoder:
    labelings = (LABELING,)
    sizing_features = (SizingFeature("size", "numeric"),)

    def decode(self, x):
        representations = []
        for row in x:
            graph = nx.Graph()
            graph.add_node(0)
            labels = {0: "root"}
            if row[0] >= 0.5:
                graph.add_edge(0, 1)
                labels[1] = "leaf"
            representations.append(
                GraphRepresentation(
                    graph,
                    {LABELING: labels},
                    np.asarray(row[:1]),
                )
            )
        return representations


def test_graph_kriging_train_and_predict():
    decoder = ExampleDecoder()
    kernel = CompositeGraphKernel(
        [
            LdWloa(1, [LABELING]),
            SizingKernel(decoder.sizing_features),
        ],
        composition="additive_interaction",
    )
    model = GraphKriging(
        decoder,
        kernel,
        hyper_opt="NoOp",
        poly="constant",
        corr="squar_exp",
        print_global=False,
        print_training=False,
        print_prediction=False,
        print_problem=False,
        print_solver=False,
        design_space=DesignSpace([IntegerVariable(0, 3)]),
    )
    x = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([[0.0], [0.5], [0.5], [0.0]])

    model.set_training_values(x, y)
    model.train()
    prediction = model.predict_values(np.array([[0.0], [3.0]]))
    variance = model.predict_variances(np.array([[0.0], [3.0]]))

    assert prediction.shape == (2, 1)
    assert variance.shape == (2, 1)
    assert np.all(np.isfinite(prediction))
    assert np.all(np.isfinite(variance))
    assert len(model.optimal_theta) == len(kernel.get_theta_parameters())
