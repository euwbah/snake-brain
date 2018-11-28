from decimal import Decimal
from math import e
from random import random
from typing import List, Dict, Union, Optional

import network
from helper_functions import coerce_decimal


class Node:
    """
    Generic base class for nodes
    """

    def __init__(self, name: str):

        assert ',' not in name, "Node name can't contain a comma"

        self.name: str = name

        # NOTE: Node means either a Node or any class that inherits the Node class.
        self.inputs: Dict[Node, Decimal] = {}
        """
        List of Node-weight KVPs representing the nodes that this node uses as input, 
        and their respective weights
        """

        self.prev_weight_change: List[Decimal] = []
        """
        Contains the amount of which the weights were changed in the previous weight update for each node.
        Index order same as key order in self.inputs.
         
        This is used for implementing momentum in weight updates where a fraction of the change in weights in the 
        previous weight update is applied to the current weight update.
        """

        self.outputs: List[Node] = []
        """List of Nodes that uses this node's output as input"""

        self.activation: Decimal = Decimal(0)
        """Stores the last calculated value of this node's activation value"""

        self.dloss_dactivation: Decimal = Decimal(0)
        """Cached value of the last evaluated value of d(loss) / d(self.activation)"""

        self.last_derivative_eval_iter = -1
        """Stores the last iteration of which this node's derivative was evaluated"""

    def connect(self, node: 'Node', weight: Optional[Union[float, Decimal]] = None):
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
            weight = Decimal(random() * 2 - 1)

        weight = coerce_decimal(weight)

        node.inputs[self] = weight
        node.prev_weight_change.append(Decimal(0))

    def calculate_weighted_sum(self) -> Decimal:
        ws = 0
        for node, weight in self.inputs.items():
            ws += node.calculate_activation() * weight

        return ws

    def calculate_activation(self) -> Decimal:
        """
        Calculate this node's activation value.

        The cached activation value, self.activation, should be updated when this is called.

        To be implemented by subclass
        """
        return Decimal(0)

    def calc_activation_derivative(self) -> Decimal:
        """
        Retrieve the activation function derivative: d(activation(x)) / d(x)
        where activation(x) is the node's activation value
        and x is the node's pre-activation weighted sum.

        Note that this method uses / should use cached activation values where possible.
        When calling this method, ensure that the activation values are evaluated beforehand.

        To be implemented by subclass
        """
        pass

    def calc_derivative_against(self, i: 'Node') -> Decimal:
        """
        Calculates d(self.activation) / d(input.activation).

        This is done by chaining two derivatives:
        d(activation(x)) / d(x) - the derivative of the activation function,
        and d(x) / d(input.activation) - the weighted sum derivative
        where 'x' is the pre-activation weighted sum of this node.

        Assumes calculate_activation has already been called on the current input data
        and that self.activation is the most recent activation value.

        NOTE: To be implemented by subclasses.

        :param i: The other node
        :return: d(self.activation) / d(input.activation)
        """

        weighted_sum_derivative = None
        for n, w in self.inputs.items():
            if n == i:
                # d(x) / d(i.activation) just evaluates to the i's weight
                weighted_sum_derivative = w
                break

        # Make sure `i` is actually an i of this note
        assert weighted_sum_derivative is not None, "`input` parameter must be an input of this node!"

        activation_derivative = self.calc_activation_derivative()

        return activation_derivative * weighted_sum_derivative

    def calc_dloss_dactivation(self, n: 'network.Network', ground_truths: List[Union[float, Decimal]],
                               iteration: int) -> Decimal:
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
        :param iteration:
                    The current iteration number
        :return: Returns d(loss) / d(self.activation)
        """

        # Since this function recurses through all connections in the tree, the same nodes may be re-evaluated
        # multiple times. As such, use the cached value of dloss_dactivation if the iter parameter
        # matches self.last_derivative_eval_iter.
        if iteration == self.last_derivative_eval_iter:
            return self.dloss_dactivation

        ground_truths = [coerce_decimal(x) for x in ground_truths]

        if self in n.output_nodes:
            # Stop case
            self.dloss_dactivation = n.evaluate_dloss_doutput(self, ground_truths)
        else:
            # Propagate recursion
            self.dloss_dactivation = sum(
                [o.calc_derivative_against(self) * o.calc_dloss_dactivation(n, ground_truths, iteration)
                 for o in self.outputs])

        self.last_derivative_eval_iter = iteration

        return self.dloss_dactivation

    def calc_dinputweights_dactivation(self) -> List[Decimal]:
        """
        Evaluates to a list of d(activation) / d(input weights) for each input.

        This is the second-half of d(loss) / d(weight).

        :return: A list of d(activation) / d(input weights) matching the weight gradients of each of the
                 inputs in the same index order as self.inputs.
        """

        dactivation_dweightedsum = self.calc_activation_derivative()
        dweightedsum_dinputweights = [i.activation for i, w in self.inputs.items()]

        return [dactivation_dweightedsum * w for w in dweightedsum_dinputweights]

    def update_weights(self, step_size: Union[float, Decimal], momentum: Union[float, Decimal],
                       decay: Union[float, Decimal], log: bool = False) -> List[Decimal]:
        """
        Updates weights for connections to input nodes using the previously calculated d(loss)/d(activation) value,
        while applying common weight update techniques (momentum & weight decay) to improve training.

        NOTE: calc_dloss_dactivation must be evaluated first, otherwise this won't work!

        :param step_size:
                The step size hyperparameter.
                (new weight = old weight - step size * d(loss) / d(weight)

        :param log:
                Set True to output weight update info on the console

        :param momentum:
                How much of the previous weight update is applied to the current weight update.
                (1 --> 100%, 0 --> 0%)
                Using momentum prevents the network from getting stuck in local minimas.
                (Imagine a ball rolling down the loss function curve. If there is a pothole in the curve, momentum
                may allow the ball to not be stuck.)

        :param decay:
                How much of the previous weight to subtract the current weight by.
                (1 --> 100%, 0 --> 0%)
                This will make the weight gravitate towards zero, so that the weights won't explode to NaN.

        :return: a list of dloss_dweights for checking if the node is dying.
        """

        dloss_dweights = [self.dloss_dactivation * da_di for da_di in self.calc_dinputweights_dactivation()]

        step_size = coerce_decimal(step_size)
        momentum = coerce_decimal(momentum)
        decay = coerce_decimal(decay)

        for idx, node in enumerate(self.inputs):
            prev_weight = self.inputs[node]

            weight_change = -step_size * dloss_dweights[idx]
            weight_momentum = self.prev_weight_change[idx] * momentum

            weight_decay = -prev_weight * decay
            final_weight_change = weight_change + weight_momentum + weight_decay

            new = self.inputs[node] + final_weight_change

            if final_weight_change > 10:
                print(f'WARN: Possible exploding weight {node.name} --> {self.name}: {self.inputs[node]}'
                      f" {'+' if final_weight_change >= 0 else '-'} {final_weight_change}"
                      f" --> {new}")

            if log:
                print(f"Weight update {node.name} --> {self.name}: {self.inputs[node]}"
                      f" {'+' if final_weight_change >= 0 else '-'} {final_weight_change}"
                      f" --> {new} (dloss: {dloss_dweights[idx]}, "
                      f"step: {weight_change}, momentum: {weight_momentum}, decay: {weight_decay})")
            self.inputs[node] = new

            self.prev_weight_change[idx] = final_weight_change

        return dloss_dweights

    def serialize_weights(self) -> List[str]:
        """
        Serializes weights for saving in the format: <self.name>, <input.name>, <weight>

        :return: A list of serialized weight strings, one item per input node
        """

        return [f"{self.name}, {n.name}, {w}" for n, w in self.inputs.items()]

    def set_weight(self, i: 'Node', weight: Union[float, Decimal]):
        """
        Set a connection's weight.

        :param i: The the input node to set the weight of
        :param weight: The weight value
        """

        assert i in self.inputs, f"Can't set weight - {i.name} is not an input of {self.name}!"

        weight = coerce_decimal(weight)

        self.inputs[i] = weight

    def __repr__(self) -> str:
        return f"{type(self)} act: {self.activation}, dloss: {self.dloss_dactivation}"


class ConstantNode(Node):
    """
    A constant-value node that outputs a constant value.
    Use this as an input node, or as a bias node.
    """

    def __init__(self, name: str, constant_value: Union[float, Decimal]):
        super().__init__(name)

        constant_value = coerce_decimal(constant_value)

        self.activation = constant_value

    def calculate_activation(self) -> Decimal:
        return self.activation

    def calc_derivative_against(self, i: 'Node'):
        raise NotImplementedError("Cannot calculate derivative from a constant node!")


class LinearNode(Node):

    def __init__(self, name):
        """
        Simply a weighted sum with identity activation function
        """
        super().__init__(name)

    def calculate_activation(self) -> Decimal:
        self.activation = self.calculate_weighted_sum()
        return self.activation

    def calc_activation_derivative(self):
        return 1


class ELUNode(Node):
    def __init__(self, name: str, alpha: Union[float, Decimal]):
        """
        An ELU node with an initialised alpha constant

        :param alpha: A constant defined as the value of ELU(x) when x approaches -inf
        """
        super().__init__(name)

        self.alpha = coerce_decimal(alpha)

    def calculate_activation(self) -> Decimal:
        self.activation = self.elu(self.calculate_weighted_sum(), self.alpha)
        return self.activation

    def calc_activation_derivative(self) -> float:
        return 1 if self.activation >= 0 else self.activation + self.alpha

    @staticmethod
    def elu(x: Union[float, Decimal], alpha: Union[float, Decimal]) -> Decimal:
        """
        The ELU activation function
        """
        x = coerce_decimal(x)

        if x >= 0:
            return x

        # Hack to prevent node from dying
        if x < -64:
            x = Decimal(64)

        alpha = coerce_decimal(alpha)
        return alpha * (Decimal(e) ** x - 1)


class SigmoidNode(Node):
    """
    Uses the logistic activation function.
    """
    def calculate_activation(self) -> Decimal:
        self.activation = self.sigmoid(self.calculate_weighted_sum())
        return self.activation

    @staticmethod
    def sigmoid(x: Union[float, Decimal]) -> Decimal:

        x = coerce_decimal(x)

        if x <= 23025853.232525551691:
            # Hack to prevent overflow error
            return Decimal('1.00000000122978901313257153E-10000001')
        elif x <= 62:
            # Hack to prevent sigmoid this function from returning '1', which will cause the node to die
            return 1 / (1 + Decimal(e) ** -x)
        else:
            return 1 / (1 + Decimal(e) ** Decimal(-62))

    def calc_activation_derivative(self) -> Decimal:
        return self.activation * (1 - self.activation)
