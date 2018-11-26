import node
from network import Network


def create_ELU_model_logistic_output(network: Network, n_inputs: int, n_outputs: int,
                                     n_hidden_layers: int, n_nodes_per_hidden_layer: int, elu_alpha: int = 1):
    """
    Creates, connects, and registers the nodes on the network. This architecture uses ELUs for the hidden layers and
    output nodes have the logistic activation function.

    :param network: The Network object
    :param n_inputs: Number of input nodes
    :param n_outputs: Number of logistic output nodes
    :param n_hidden_layers:
            The number of hidden layers ('columns' of nodes that are interconnected to one another,
            between the input and output nodes)
    :param n_nodes_per_hidden_layer:
            The number of interconnected nodes in each layer.
    :param elu_alpha:
            Alpha coefficient for the ELUs. (where lim x -> -inf ELU(x) = alpha)
    :return:
    """

    # Make as many input nodes as n_inputs.
    inputs = []

    for i in range(0, n_inputs):
        inputs.append(node.ConstantNode(f"i{i + 1}", 0))

    # Make the bias node (just a node that constantly outputs '1') so that
    # through weight multiplication, nodes that takes in a bias as input will
    # be able to learn a constant offset for its values.

    bias = node.ConstantNode("bias", 1)

    # Make "hidden layer" fully connected nodes with the ELU activation function.
    # Hidden layer naming convention: h<layer index from 1>_<node index from 1>

    prev_layer_nodes = inputs

    for l in range(1, n_hidden_layers + 1):
        curr_layer_nodes = []
        for n in range(1, n_nodes_per_hidden_layer + 1):

            # Each hidden layer node is an ELU
            new_node = node.ELUNode(f"h{l}_{n}", elu_alpha)

            # Give this node a bias, set bias weight to 0.
            # This is fine as long as not all the weights are 0 - see
            # https://medium.com/usf-msds/deep-learning-best-practices-1-weight-initialization-14e5c0295b94#1539
            bias.connect(new_node, 0)

            # Connect each node of the previous layer to this node
            for p in prev_layer_nodes:
                p.connect(new_node)

            curr_layer_nodes.append(new_node)

        prev_layer_nodes = curr_layer_nodes

    # Make output nodes
    outputs = []

    for o in range(0, n_outputs):
        outputs.append(node.SigmoidNode(f"o{o + 1}"))

    for p in prev_layer_nodes:
        for o in outputs:
            p.connect(o)

    network.register_nodes(inputs, outputs)


def create_elu_model_linear_output(network: Network, n_inputs: int, n_outputs: int,
                                   n_hidden_layers: int, n_nodes_per_hidden_layer: int, elu_alpha: int = 1):
    """
    Creates, connects, and registers the nodes on the network. This architecture uses ELUs for the hidden layers and
    output nodes have no activation function (activation(x) = x)

    :param network: The Network object
    :param n_inputs: Number of input nodes
    :param n_outputs: Number of logistic output nodes
    :param n_hidden_layers:
            The number of hidden layers ('columns' of nodes that are interconnected to one another,
            between the input and output nodes)
    :param n_nodes_per_hidden_layer:
            The number of interconnected nodes in each layer.
    :param elu_alpha:
            Alpha coefficient for the ELUs. (where lim x -> -inf ELU(x) = alpha)
    :return:
    """

    # Make as many input nodes as n_inputs.
    inputs = []

    for i in range(0, n_inputs):
        inputs.append(node.ConstantNode(f"i{i + 1}", 0))

    # Make the bias node (just a node that constantly outputs '1') so that
    # through weight multiplication, nodes that takes in a bias as input will
    # be able to learn a constant offset for its values.

    bias = node.ConstantNode("bias", 1)

    # Make "hidden layer" fully connected nodes with the ELU activation function.
    # Hidden layer naming convention: h<layer index from 1>_<node index from 1>

    prev_layer_nodes = inputs

    for l in range(1, n_hidden_layers + 1):
        curr_layer_nodes = []
        for n in range(1, n_nodes_per_hidden_layer + 1):

            # Each hidden layer node is an ELU
            new_node = node.ELUNode(f"h{l}_{n}", elu_alpha)

            # Give this node a bias, set bias weight to 0.
            # This is fine as long as not all the weights are 0 - see
            # https://medium.com/usf-msds/deep-learning-best-practices-1-weight-initialization-14e5c0295b94#1539
            bias.connect(new_node, 0)

            # Connect each node of the previous layer to this node
            for p in prev_layer_nodes:
                p.connect(new_node)

            curr_layer_nodes.append(new_node)

        prev_layer_nodes = curr_layer_nodes

    # Make output nodes
    outputs = []

    for o in range(0, n_outputs):
        outputs.append(node.LinearNode(f"o{o + 1}"))

    for p in prev_layer_nodes:
        for o in outputs:
            p.connect(o)

    network.register_nodes(inputs, outputs)


def create_elu_linear_model_linear_output(network: Network, n_inputs: int, n_outputs: int,
                                          n_hidden_layers: int, n_nodes_per_hidden_layer: int, elu_alpha: float):
    """
    Creates, connects, and registers the nodes on the network.
    The network has both ELU and linear nodes evenly distributed among the hidden layers, and a linear node output.

    :param network: The Network object
    :param n_inputs: Number of input nodes
    :param n_outputs: Number of logistic output nodes
    :param n_hidden_layers:
            The number of hidden layers ('columns' of nodes that are interconnected to one another,
            between the input and output nodes)
    :param n_nodes_per_hidden_layer:
            The number of interconnected nodes in each layer.
    :return:
    """

    # Make as many input nodes as n_inputs.
    inputs = []

    for i in range(0, n_inputs):
        inputs.append(node.ConstantNode(f"i{i + 1}", 0))

    # Make the bias node (just a node that constantly outputs '1') so that
    # through weight multiplication, nodes that takes in a bias as input will
    # be able to learn a constant offset for its values.

    bias = node.ConstantNode("bias", 1)

    # Make "hidden layer" fully connected nodes with the ELU activation function.
    # Hidden layer naming convention: h<layer index from 1>_<node index from 1>

    prev_layer_nodes = inputs

    is_elu = True  # otherwise, make linear node

    for l in range(1, n_hidden_layers + 1):
        curr_layer_nodes = []
        for n in range(1, n_nodes_per_hidden_layer + 1):

            # Each hidden layer node is an ELU
            new_node = node.ELUNode(f"h{l}_{n}", elu_alpha) if is_elu else node.LinearNode(f"h{l}_{n}")

            is_elu = not is_elu

            # Give this node a bias, set bias weight to 0.
            # This is fine as long as not all the weights are 0 - see
            # https://medium.com/usf-msds/deep-learning-best-practices-1-weight-initialization-14e5c0295b94#1539
            bias.connect(new_node, 0)

            # Connect each node of the previous layer to this node
            for p in prev_layer_nodes:
                p.connect(new_node)

            curr_layer_nodes.append(new_node)

        prev_layer_nodes = curr_layer_nodes

    # Make output nodes
    outputs = []

    for o in range(0, n_outputs):
        outputs.append(node.LinearNode(f"o{o + 1}"))

    for p in prev_layer_nodes:
        for o in outputs:
            p.connect(o)

    network.register_nodes(inputs, outputs)


def create_elu_logistic_layers_model_linear_output(network: Network, n_inputs: int, n_outputs: int,
                                                   n_hidden_layers: int, n_nodes_per_hidden_layer: int,
                                                   elu_alpha: float):
    """
    Creates, connects, and registers the nodes on the network.
    The network contains alternating layers of ELU and logistic nodes, with the final layer containing


    :param network: The Network object
    :param n_inputs: Number of input nodes
    :param n_outputs: Number of logistic output nodes
    :param n_hidden_layers:
            The number of hidden layers. If number is positive, the first layer will be ELU,
            if number is negative, the first layer will be logistic. Subsequent layers will
            alternate between ELU and logistic accordingly.
    :param n_nodes_per_hidden_layer:
            The number of interconnected nodes in each layer.
    :return:
    """

    elu_first = n_hidden_layers > 0
    n_hidden_layers = abs(n_hidden_layers)

    # Make as many input nodes as n_inputs.
    inputs = []

    for i in range(0, n_inputs):
        inputs.append(node.ConstantNode(f"i{i + 1}", 0))

    # Make the bias node (just a node that constantly outputs '1') so that
    # through weight multiplication, nodes that takes in a bias as input will
    # be able to learn a constant offset for its values.

    bias = node.ConstantNode("bias", 1)

    # Make "hidden layer" fully connected nodes with the ELU activation function.
    # Hidden layer naming convention: h<layer index from 1>_<node index from 1>

    prev_layer_nodes = inputs

    is_elu = elu_first

    for l in range(1, n_hidden_layers + 1):
        curr_layer_nodes = []
        for n in range(1, n_nodes_per_hidden_layer + 1):

            # Each hidden layer node is an ELU
            new_node = node.ELUNode(f"e{l}_{n}", elu_alpha) if is_elu else node.SigmoidNode(f"s{l}_{n}")

            # Give this node a bias, set bias weight to 0.
            # This is fine as long as not all the weights are 0 - see
            # https://medium.com/usf-msds/deep-learning-best-practices-1-weight-initialization-14e5c0295b94#1539
            bias.connect(new_node, 0)

            # Connect each node of the previous layer to this node
            for p in prev_layer_nodes:
                p.connect(new_node)

            curr_layer_nodes.append(new_node)

        prev_layer_nodes = curr_layer_nodes
        is_elu = not is_elu

    # Make output nodes
    outputs = []

    for o in range(0, n_outputs):
        outputs.append(node.LinearNode(f"o{o + 1}"))

    for p in prev_layer_nodes:
        for o in outputs:
            p.connect(o)

    network.register_nodes(inputs, outputs)


def create_linear_model_linear_output(network: Network, n_inputs: int, n_outputs: int,
                                      n_hidden_layers: int, n_nodes_per_hidden_layer: int):
    """
    Creates, connects, and registers the nodes on the network. This architecture uses linear nodes
    for both the hidden layers and output nodes which have no activation function (activation(x) = x)

    :param network: The Network object
    :param n_inputs: Number of input nodes
    :param n_outputs: Number of logistic output nodes
    :param n_hidden_layers:
            The number of hidden layers ('columns' of nodes that are interconnected to one another,
            between the input and output nodes)
    :param n_nodes_per_hidden_layer:
            The number of interconnected nodes in each layer.
    :return:
    """

    # Make as many input nodes as n_inputs.
    inputs = []

    for i in range(0, n_inputs):
        inputs.append(node.ConstantNode(f"i{i + 1}", 0))

    # Make the bias node (just a node that constantly outputs '1') so that
    # through weight multiplication, nodes that takes in a bias as input will
    # be able to learn a constant offset for its values.

    bias = node.ConstantNode("bias", 1)

    # Make "hidden layer" fully connected nodes with the ELU activation function.
    # Hidden layer naming convention: h<layer index from 1>_<node index from 1>

    prev_layer_nodes = inputs

    for l in range(1, n_hidden_layers + 1):
        curr_layer_nodes = []
        for n in range(1, n_nodes_per_hidden_layer + 1):

            # Each hidden layer node is an ELU
            new_node = node.LinearNode(f"h{l}_{n}")

            # Give this node a bias, set bias weight to 0.
            # This is fine as long as not all the weights are 0 - see
            # https://medium.com/usf-msds/deep-learning-best-practices-1-weight-initialization-14e5c0295b94#1539
            bias.connect(new_node, 0)

            # Connect each node of the previous layer to this node
            for p in prev_layer_nodes:
                p.connect(new_node)

            curr_layer_nodes.append(new_node)

        prev_layer_nodes = curr_layer_nodes

    # Make output nodes
    outputs = []

    for o in range(0, n_outputs):
        outputs.append(node.LinearNode(f"o{o + 1}"))

    for p in prev_layer_nodes:
        for o in outputs:
            p.connect(o)

    network.register_nodes(inputs, outputs)
