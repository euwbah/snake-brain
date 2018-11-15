from math import exp
from random import random
from typing import List, Tuple, Dict

import network


class Node:
    """
    Generic base class for nodes
    """

    def __init__(self, name: str):

        assert ',' not in name, "Node name can't contain a comma"

        self.name: str = name

        # NOTE: Node means either a Node or any class that inherits the Node class.
        self.inputs: Dict[Node, float] = {}
        """
        List of Node-weight tuples representing the nodes that this node uses as input, 
        and their respective weights
        """

        self.outputs: List[Node] = []
        """List of Nodes that uses this node's output as input"""

        self.activation: float = 0
        """Stores the last calculated value of this node's activation value"""

        self.dloss_dactivation = 0
        """Cached value of the last evaluated value of d(loss) / d(self.activation)"""

        self.last_derivative_eval_iter = -1
        """Stores the last iteration of which this node's derivative was evaluated"""

    def connect(self, node: 'Node', weight: float = None):
        """
        Connect self's output to node's input

        :param node: The node which receives this node's output as input
        :param weight: Optional preset weight. If not provided, evaluates to a random number in the interval [-1, 1)
        """
        self.outputs.append(node)

        # NOTE: This can't be expressed as a default parameter as the pseudo random generator uses
        #       time as the sole independent variable, and there is only one method being created, hence
        #       all the instances of this node will be initialized to the same value, making the neural
        #       network essentially a linear regression model.
        if weight is None:
            weight = random() * 2 - 1

        node.inputs[self] = weight

    def calculate_weighted_sum(self) -> float:
        ws = 0
        for node, weight in self.inputs.items():
            ws += node.calculate_activation() * weight

        return ws

    def calculate_activation(self) -> float:
        """
        Calculate this node's activation value.

        The cached activation value, self.activation, should be updated when this is called.

        To be implemented by subclass

        :return:
        """
        return 0

    def calc_activation_derivative(self) -> float:
        """
        Retrieve the activation function derivative: d(activation(x)) / d(x)
        where activation(x) is the node's activation value
        and x is the node's pre-activation weighted sum.

        Note that this method uses / should use cached activation values where possible.
        When calling this method, ensure that the activation values are evaluated beforehand.

        To be implemented by subclass
        :return:
        """
        pass

    def calc_derivative_against(self, input: 'Node') -> float:
        """
        Calculates d(self.activation) / d(input.activation).

        This is done by chaining two derivatives:
        d(activation(x)) / d(x) - the derivative of the activation function,
        and d(x) / d(input.activation) - the weighted sum derivative
        where 'x' is the pre-activation weighted sum of this node.

        Assumes calculate_activation has already been called on the current input data
        and that self.activation is the most recent activation value.

        NOTE: To be implemented by subclasses.

        :param input: The other node
        :return: d(self.activation) / d(input.activation)
        """

        weighted_sum_derivative = None
        for n, w in self.inputs.items():
            if n == input:
                # d(x) / d(input.activation) just evaluates to the input's weight
                weighted_sum_derivative = w
                break

        # Make sure `input` is actually an input of this note
        assert weighted_sum_derivative is not None, "`input` parameter must be an input of this node!"

        activation_derivative = self.calc_activation_derivative()

        return activation_derivative * weighted_sum_derivative

    def calc_dloss_dactivation(self, n: 'network.Network', ground_truths: List[float], iter: int) -> float:
        """
        Use recursion to "backpropagate" d(loss) / d(self.activation) for all connected nodes in the network.
        Start the recursion off by calling this on the INPUT nodes.

        This is the first-half of d(loss) / d(weight)

        :param n:
                    The network object containing the output_nodes which are the stop cases for the recursion,
                    and the evaluate_dloss_doutput function which evaluates the derivative of loss-against-output-nodes
                    for the stop cases.
        :param ground_truths:
                    The list of ground truth values - same as the one passed in to Network.evaluate_loss()
        :param iter:
                    The current iteration number
        :return: Returns d(loss) / d(self.activation)
        """

        # Since this function recurses through all connections in the tree, the same nodes may be re-evaluated
        # multiple times. As such, use the cached value of dloss_dactivation if the iter parameter
        # matches self.last_derivative_eval_iter.
        if iter == self.last_derivative_eval_iter:
            return self.dloss_dactivation

        if self in n.output_nodes:
            # Stop case
            self.dloss_dactivation = n.evaluate_dloss_doutput(self, ground_truths)
        else:
            # Propagate recursion
            self.dloss_dactivation = sum([o.calc_derivative_against(self) * o.calc_dloss_dactivation(n, ground_truths, iter)
                                          for o in self.outputs])

        self.last_derivative_eval_iter = iter

        return self.dloss_dactivation

    def calc_dinputweights_dactivation(self) -> List[float]:
        """
        Evaluates to a list of d(activation) / d(input weights) for each input.

        This is the second-half of d(loss) / d(weight).

        :return: A list of d(activation) / d(input weights) matching the weight gradients of each of the
                 inputs in the same index order as self.inputs.
        """

        dactivation_dweightedsum = self.calc_activation_derivative()
        dweightedsum_dinputweights = [i.activation for i, w in self.inputs.items()]

        return [dactivation_dweightedsum * w for w in dweightedsum_dinputweights]

    def update_weights(self, step_size: float, log: bool = False):
        """
        Updates weights for connections to input nodes using the previously calculated d(loss)/d(activation) value.

        NOTE: calc_dloss_dactivation must be evaluated first, otherwise this won't work!

        :param step_size:
                The step size hyperparameter.
                (new weight = old weight - step size * d(loss) / d(weight)
        :param log:
                Set True to output weight update info on the console
        """

        dloss_dweights = [self.dloss_dactivation * da_di for da_di in self.calc_dinputweights_dactivation()]

        for idx, node in enumerate(self.inputs):
            new = self.inputs[node] - step_size * dloss_dweights[idx]
            if log:
                print(f"Updated weight of {node.name}: {self.inputs[node]} --> {new} (dloss: {dloss_dweights[idx]})")
            self.inputs[node] = new

    def serialize_weights(self) -> List[str]:
        """
        Serializes weights for saving in the format: <self.name>, <input.name>, <weight>

        :return: A list of serialized weight strings, one item per input node
        """

        return [f"{self.name}, {n.name}, {w}" for n, w in self.inputs.items()]

    def set_weight(self, input: 'Node', weight: float):
        """
        Set a connection's weight.

        :param input: The the input node to set the weight of
        :param weight: The weight value
        """

        assert input in self.inputs, f"Can't set weight - {node.name} is not an input of {self.name}!"

        self.inputs[input] = weight


class ConstantNode(Node):
    """
    A constant-value node that outputs a constant value.
    Use this as an input node, or as a bias node.
    """

    def __init__(self, name: str, constant_value: float):
        super().__init__(name)

        self.activation = constant_value

    def calculate_activation(self) -> float:
        return self.activation

    def calc_derivative_against(self, input: 'Node'):
        raise NotImplementedError("Cannot calculate derivative from a constant node!")


class ELUNode(Node):
    def __init__(self, name: str, alpha: float):
        """
        An ELU node with an initialised alpha constant

        :param alpha: A constant defined as the value of ELU(x) when x approaches -inf
        """
        super().__init__(name)

        self.alpha = alpha

    def calculate_activation(self) -> float:
        self.activation = self.elu(self.calculate_weighted_sum(), self.alpha)
        return self.activation

    def calc_activation_derivative(self) -> float:
        return 1 if self.activation >= 0 else self.activation + self.alpha

    @staticmethod
    def elu(x: float, alpha: float) -> float:
        """
        The ELU activation function
        """
        return x if x >= 0 else alpha * (exp(x) - 1)


class SigmoidNode(Node):
    def calculate_activation(self) -> float:
        self.activation = self.sigmoid(self.calculate_weighted_sum())
        return self.activation

    @staticmethod
    def sigmoid(x: float) -> float:
        # Hack to prevent overflows
        if 709.782 >= x >= -709.782:
            return 1 / (1 + exp(-x))
        else:
            return 1 / (1 + exp(709.782 if x > 0 else -709.782))

    def calc_activation_derivative(self):
        return self.activation * (1 - self.activation)
