import node
from data import Data
from network import Network


def test_forward_propagation(network: Network, data: Data):

    i1 = node.ConstantNode("i1", data[0][0][0])
    i2 = node.ConstantNode("i2", data[0][0][1])

    h1_1 = node.ELUNode("h1_1", 1)
    h1_2 = node.ELUNode("h1_2", 1)
    i1.connect(h1_1, 1)
    i1.connect(h1_2, 0)
    i2.connect(h1_1, 0)
    i2.connect(h1_2, 1)

    h2_1 = node.ELUNode("h2_1", 1)
    h2_2 = node.ELUNode("h2_2", 1)
    h1_1.connect(h2_1, 1)
    h1_1.connect(h2_2, 0)
    h1_2.connect(h2_1, 0)
    h1_2.connect(h2_2, 1)

    o1 = node.SigmoidNode("o1")
    h2_1.connect(o1, 1)
    h2_2.connect(o1, 0)

    network.register_nodes([i1, i2], [o1])

    print(network.forward_propagate())
    print(network.evaluate_loss(data[0][1]))


def test_evaluate_dloss_dactivation(network: Network, data: Data):
    ground_truths = data[0][1]
    network.evaluate_gradients(ground_truths)

