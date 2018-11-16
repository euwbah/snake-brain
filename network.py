from typing import List, Dict
import re

WEIGHT_RE = re.compile(r'\s*(?P<node>[^,]+)\s*,\s*(?P<input>[^,]+)\s*,\s*(?P<weight>.+?)\s*')
ITER_RE = re.compile(r'iter:\s*(?P<iter>[0-9]+)\s*')


class Network:
    """
    The neural network graph of nodes
    """

    def __init__(self):
        from node import Node, ConstantNode
        self.input_nodes: List[ConstantNode] = []
        self.output_nodes: List[Node] = []

        self.nodes: Dict[str, Node] = {}

        self.iter: int = 0
        """
        Iteration counter that increments after each call to train_iter().
        
        This value persists across weight saves/loads. 
        """

    def register_nodes(self, input_nodes: List['ConstantNode'], output_nodes: List['Node']):
        """
        Populates the Network structure with a dictionary of nodes by recursion of
        the node's inputs starting from the output nodes.

        Also ensures that nodes don't share an identical name.

        :param input_nodes: A list of input nodes that the network can map input data to
        :param output_nodes: A list of output nodes for activation calculation & ground-truth error evaluation
        """
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes

        def register_node_and_inputs(node: 'Node'):

            assert node.name not in self.nodes, \
                f"Node name conflicted: {node.name}. Ensure that node names are unique and there are no " \
                f"cyclically connected nodes."

            self.nodes[node.name] = node
            for i, _ in node.inputs.items():
                if i.name not in self.nodes or self.nodes[i.name] != i:
                    # Nodes in one layer may share the same input node, a node already registered
                    # does not need to be registered again and should not count as a node conflict.
                    register_node_and_inputs(i)

        # For each output node, add it to the node registry, for each of its inputs, add it to the registry, etc...
        for n in output_nodes:
            register_node_and_inputs(n)

    def assign_inputs(self, input_values: List[float]):
        """
        Set the constant activation value for the network's input nodes to the values in input_values

        :param input_values:
                A list of values to set the input_nodes activations to. Values are set according to index order.
                i.e. input_nodes[idx].activation = input_values[idx]
        :return:
        """
        assert len(input_values) == len(self.input_nodes), "len(input_values) need to match len(Network.input_nodes)"

        for idx, i in enumerate(self.input_nodes):
            i.activation = input_values[idx]

    def forward_propagate(self) -> List[float]:
        """
        Perform "forward propagation" recursively.

        :return: a list of activations corresponding to the activation values of the output nodes on the final layer.
        The values correspond in the same order as self.output_nodes
        """
        return [o.calculate_activation() for o in self.output_nodes]

    def evaluate_loss(self, ground_truths: List[float]) -> float:
        """
        Evaluates the loss score using the Mean-Square Error function.
        Note that this function uses the previously cached activation values. Forward propagation must
        be performed before calling this function

        :param ground_truths:
        :return:
        """
        assert len(ground_truths) == len(self.output_nodes), \
            "Number of ground truths need to match number of output nodes!"

        square_error = sum([(self.output_nodes[i].activation - g) ** 2 for i, g in enumerate(ground_truths)])
        mean_square_error = square_error / len(self.output_nodes)
        return mean_square_error

    def evaluate_dloss_doutput(self, output_node: 'Node', ground_truths: List[float]) -> float:
        """
        Evaluates d(loss) / d(output_node.activation) assuming the loss function used is MSE

        :param output_node:
        :param ground_truths: The same ground truth values that was passed to evaluate_loss
        :return: d(loss) / d(output_node.activation)
        """

        assert output_node in self.output_nodes, \
            "output_node param must be registered as an output node of the network!"
        ground_truth = ground_truths[self.output_nodes.index(output_node)]

        return 2 * (output_node.activation - ground_truth)

    def evaluate_gradients(self, ground_truths: List[float]):
        """
        Calls calc_dloss_dactivation on all the input nodes to recursively evaluate the d(loss) / d(activation)
        derivatives on all nodes.

        NOTE: forward_propagate must be called first!
        """

        assert len(ground_truths) == len(self.output_nodes), \
            "Number of ground truths need to match number of output nodes!"

        for i in self.input_nodes:
            i.calc_dloss_dactivation(self, ground_truths, self.iter)

    def update_weights(self, step_size: float, log: bool = False) -> (float, float):
        """
        Update weights of all registered nodes.

        NOTE: evaluate_gradients must be called first!

        :param log: Set True to show information of each weight update
        :return (average dloss_dweight, max dloss_dweight) amongst all the nodes
        """

        dlosses = []
        max_dloss = 0

        for node in self.nodes.values():
            gradients = node.update_weights(step_size, log)
            if len(gradients) > 0:
                dlosses += gradients
                max_gradient = max(gradients)
                if max_gradient > max_dloss:
                    max_dloss = max_gradient

        return sum(dlosses) / len(dlosses), max_dloss

    def save_weights(self, path: str):
        """
        Saves all weights & other meta information in the file at specified path

        :param path: The path of the file
        """
        with open(path, "w") as f:

            f.write(f"iter: {self.iter}\n")

            for node in self.nodes.values():
                for weight_info in node.serialize_weights():
                    f.write(weight_info + "\n")

    def load_weights(self, path: str):
        """
        Load connection weights & other meta information from the file at the specified path.
        Make sure that the nodes are connected and registered first, and that
        the network model architecture that was used to save the weights file
        is the same as this network.

        :param path: The path of the file
        """

        contents = ""

        with open(path, "r") as f:
            contents = f.read()

        lines = contents.split('\n')

        for line in lines:

            iter_match = ITER_RE.fullmatch(line)
            if iter_match is not None:
                self.iter = int(iter_match.group("iter"))
                continue

            weight_match = WEIGHT_RE.fullmatch(line)
            if weight_match is not None:
                node_name = weight_match.group("node")
                input_name = weight_match.group("input")
                weight_str = weight_match.group("weight")

                assert node_name in self.nodes, f"Node '{node_name}' not registered in the network!"
                assert input_name in self.nodes, f"Node '{input_name}' not registered in the network!"

                try:
                    weight = float(weight_str)
                    self.nodes[node_name].set_weight(self.nodes[input_name], weight)
                except ValueError:
                    raise AssertionError(f"Invalid weight file format - weight '{weight}' is not a float")

                continue

    def train_iter(self, inputs: List[float], ground_truths: List[float], step_size: float, verbose: bool = False) \
            -> (float, float, float):
        """
        Train a single iteration of the network.
        :param inputs:
        :param ground_truths:
        :param step_size:
        :param verbose:
        :return (training loss, avg. dloss_dweight, max dloss_dweight)
        """

        if verbose:
            print(f"Total iterations trained: {self.iter}")
            print(f"Assigning inputs: {inputs}")
        self.assign_inputs(inputs)

        output_activations = self.forward_propagate()
        if verbose:
            print(f"Output activations: {output_activations}")

        loss = self.evaluate_loss(ground_truths)

        self.evaluate_gradients(ground_truths)
        avg_dloss, max_dloss = self.update_weights(step_size, verbose)

        self.iter += 1

        return loss, avg_dloss, max_dloss

    def validate(self, inputs: List[float], ground_truths: List[float], verbose: bool = False) -> float:
        """
        Does a forward pass and returns the loss. No weight update.

        :param inputs:
        :param ground_truths:
        :param verbose:
        :return: the loss score
        """

        if verbose:
            print(f"Assigning inputs: {inputs}")
        self.assign_inputs(inputs)

        output_activations = self.forward_propagate()
        if verbose:
            print(f"Output activations: {output_activations}")

        loss = self.evaluate_loss(ground_truths)

        return loss

    def predict(self, inputs: List[float]) -> List[float]:
        """
        Set inputs & get the network's output

        :param inputs: Input values
        :return: Output values
        """

        self.assign_inputs(inputs)
        return self.forward_propagate()
