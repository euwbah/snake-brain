from typing import Tuple, List
from random import random, randint, choice

from network import Network
import node

# Data type alias
Data = List[Tuple[List[float], List[float]]]

def generate_data(size: int, min: float = -100, max: float = 100, max_multiple: int = 20) -> Data:
    """

    Generates sample data of the following two categories in equal proportion:
        1. [x, y] -> [1] where x is a positive/negative integer multiple of y
        2. [a, b] -> [0] where a b are random numbers of which abs(a - round_multiple(a, b)) is at least 5% of b.

        where a, b, x, y are limited to the range from `min` to `max`.

    :param size:
                The number of datasets to create.
    :param min:
                Minimum values of the input dataset
    :param max:
                Maximum values of the input dataset
    :param max_multiple:
                The highest value of abs(x / y) generated
    :return: A list of (input vector, ground truth vector) tuples
    """

    data = []
    generate_correct = True

    for i in range(0, size):
        if generate_correct:
            x = random() * (max - min) - min
            y = x / randint(1, max_multiple) * choice([-1, 1])
            data.append(([x, y], [1]))
        else:
            while True:
                a = random() * (max - min) + min
                b = random() * (max - min) + min

                # Make sure values of a and b are not within 5% range of being
                # the correct answer relative to the value of b.
                if abs(a - (b * round(a / b))) / b < 0.05:
                    continue

                data.append(([a, b], [0]))
                break

        generate_correct = not generate_correct

    return data

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

def create_network_model(network: Network, n_hidden_layers: int, n_nodes_per_hidden_layer: int):
    """
    Creates, connects, and registers the nodes on the network for the task of identifying if
    i1 is perfectly divisible by i2.

    NOTE: If the network model architecture is changed, the saved weights file from the previous model will not work.
          Ensure backups are made for this function.

    :param network: The Network object
    :param n_hidden_layers:
            The number of hidden layers ('columns' of nodes that are interconnected to one another,
            between the input and output nodes)
    :param n_nodes_per_hidden_layer:
            The number of interconnected nodes in each layer.
    :return:
    """

    # Make input nodes (constant value will be assigned in Network.assign_inputs)
    i1 = node.ConstantNode("i1", 0)
    i2 = node.ConstantNode("i2", 0)

    # Make the bias node (just a node that constantly outputs '1') so that
    # through weight multiplication, nodes that takes in a bias as input will
    # be able to learn a constant offset for its values.

    bias = node.ConstantNode("bias", 1)

    # Make "hidden layer" fully connected nodes with the ELU activation function.
    # Hidden layer naming convention: h<layer index from 1>_<node index from 1>

    prev_layer_nodes = [i1, i2]

    for l in range(1, n_hidden_layers + 1):
        curr_layer_nodes = []
        for n in range(1, n_nodes_per_hidden_layer + 1):

            # Each hidden layer node is an ELU with an alpha value of 5
            new_node = node.ELUNode(f"h{l}_{n}", 5)

            # Give this node a bias
            bias.connect(new_node)

            # Connect each node of the previous layer to this node
            for p in prev_layer_nodes:
                p.connect(new_node)

            curr_layer_nodes.append(new_node)

        prev_layer_nodes = curr_layer_nodes

    # Make output node
    o1 = node.SigmoidNode("o1")
    for p in prev_layer_nodes:
        p.connect(o1)

    network.register_nodes([i1, i2], [o1])


def train_one_epoch(network: Network, data: Data, step_size: float, save_file_prefix: str, verbose: bool = False,
                    pause_after_iter: bool = False):
    """
    Train the network for 1 epoch (1 forward and backward prop through each item in the data set.)

    :param network: The Network object
    :param data: Training data
    :param step_size: The multiplier of the d(loss)/d(weight) derivative the weights are updated by.
    :param save_file_prefix: The file name (and folder path) prefix used when saving the weight file.
    :param verbose: Set True to show additional verbose logs
    :return:
    """

    for inputs, ground_truths in data:
        network.train_iter(inputs, ground_truths, step_size, verbose)
        if pause_after_iter:
            input()

    network.save_weights(f"{save_file_prefix}.txt")

def train_model(model_name: str):
    """
    Generate

    :return:
    """
    data = generate_data(100)
    network = Network()
    create_network_model(network, 5, 10)
    train_one_epoch(network, data, 0.0005, model_name, False)

if __name__ == "__main__":
    train_model("test1")

    exit(0)
